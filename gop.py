# -*- coding: utf-8 -*-
"""
Avaliação de pronúncia (local/offline) com:
- PER (com S/D/I, por classe e por posição sem recontagem),
- Intelligibility = 1 − PER,
- GOP com rivais por classe e thresholds por fonema (otimizado por lote),
- WAR (legado + exact + tolerant + hybrid; por comprimento e ponderado),
- AER por traços (com deleções/ins. e 'round'; correção de ditongos + canonização),
- IC-bootstrap (vetorizado em NumPy),
- Fluency avançado (VAD + pausas; WPM; articulation rate; MLR; variabilidade; fillers),
- Confusion list por palavra,
- Normalização de símbolos (ʧ→tʃ, ʤ→dʒ, ɡ→g, r→ɹ, ɚ→ɝ, opç. ɜ→ə).

Modelo: vitouphy/wav2vec2-xls-r-300m-timit-phoneme (CTC).
Tudo roda local (sem rede).

Requisitos principais:
    pip install torch transformers soundfile numpy g2p_en scipy

Acelerações opcionais (recomendadas):
    pip install rapidfuzz numba torchaudio  # (torchaudio opcional; rapidfuzz/numba aceleram Levenshtein)

Notas de performance:
- Singleton do modelo + autocast(CUDA/MPS) + torch.compile (se disponível).
- Cache @lru_cache para G2P por palavra (com casefold).
- Levenshtein usando RapidFuzz (C++) quando disponível; fallback para DP em Python.
- Bootstrap de PER 100% vetorizado (NumPy).
- Resample com scipy.signal.resample_poly (rápido); fallback para torchaudio se preferir.
- GOP em lote: pré-computa rivais max por classe (all/vowel/cons) por segmento.
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from functools import lru_cache

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from g2p_en import G2p

# Aceleração opcional de Levenshtein
try:
    from rapidfuzz.distance import Levenshtein as RFLev
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# Opcional: Numba (não utilizada diretamente no fallback atual)
try:
    from numba import njit  # noqa: F401
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

import torch
from transformers import AutoProcessor, AutoModelForCTC


# ========================= Util: texto =========================

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize_words(s: str) -> List[str]:
    s = re.sub(r"[^\w\s']", " ", (s or ""))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return [t for t in s.split() if t]


# ========================= Áudio (resample otimizado) =========================

def load_audio_16k_mono(path: str) -> np.ndarray:
    """
    Leitura com soundfile e resample rápido com resample_poly (≈3–5x mais rápido que librosa.resample).
    """
    audio, sr = sf.read(path, always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != 16000:
        g = int(np.gcd(sr, 16000))
        up = 16000 // g
        down = sr // g
        audio = resample_poly(audio, up, down).astype(np.float32)
    return audio.astype(np.float32)


# ========================= G2P e mapeamento =========================

_g2p = G2p()
_ARPA_RE = re.compile(r"[A-Z]+[0-2]?")  # tokens ARPAbet (com stress opcional)

@lru_cache(maxsize=100_000)
def _g2p_arpabet_tokens_cached(word_casefold: str) -> List[str]:
    toks = [t for t in _g2p(word_casefold) if t.strip()]
    return [t for t in toks if _ARPA_RE.fullmatch(t)]

def g2p_arpabet_tokens(word: str) -> List[str]:
    # cache por palavra em lower/casefold para maximizar acertos
    return _g2p_arpabet_tokens_cached(word.casefold())

def strip_stress(ph: str) -> str:
    return re.sub(r"[0-2]$", "", ph)

# Candidatos IPA/TIMIT por ARPAbet (ordem de preferência)
_ARPA_TO_IPA_CANDIDATES: Dict[str, List[str]] = {
    # Vogais
    "AA": ["ɑ", "aa"], "AE": ["æ", "ae"], "AH": ["ʌ", "ah"], "AO": ["ɔ", "ao"],
    "EH": ["ɛ", "eh"], "IH": ["ɪ", "ih"], "IY": ["i", "iy"], "UH": ["ʊ", "uh"], "UW": ["u", "uw"],
    "ER": ["ɝ", "ɚ", "er"], "EY": ["eɪ", "ey"], "AY": ["aɪ", "ay"], "OW": ["oʊ", "ow"],
    "OY": ["ɔɪ", "oy"], "AW": ["aʊ", "aw"],
    # Consoantes
    "P": ["p"], "B": ["b"], "T": ["t"], "D": ["d"], "K": ["k"], "G": ["ɡ", "g"],
    "CH": ["tʃ", "ch"], "JH": ["ʤ", "dʒ", "jh"], "SH": ["ʃ", "sh"], "ZH": ["ʒ", "zh"],
    "S": ["s"], "Z": ["z"], "F": ["f"], "V": ["v"],
    "TH": ["θ", "th"], "DH": ["ð", "dh"],
    "M": ["m"], "N": ["n"], "NG": ["ŋ", "ng"],
    "L": ["l"], "R": ["ɹ", "r"], "Y": ["j", "y"], "W": ["w"], "HH": ["h", "hh"],
    # Pausas
    "SIL": ["sil"], "SPN": ["spn"], "NSN": ["nsn"],
}

def build_arpabet_to_model_vocab_map(model_vocab: Dict[str, int]) -> Dict[str, str]:
    """Cria um mapeamento ARPAbet→símbolos do vocabulário do modelo (IPA/TIMIT).

        A função percorre uma lista de candidatos por símbolo ARPAbet (ex.: `AE→["æ","ae"]`,
        `CH→["tʃ","ch"]`, `ER→["ɝ","ɚ","er"]`) e escolhe **o primeiro candidato que existir**
        no vocabulário do modelo (`model_vocab`). O retorno é um dicionário cujo *key* é o
        símbolo ARPAbet **sem marca de acento** (stress) e o *value* é o token equivalente
        presente no vocabulário do modelo.

        Observações importantes:
          * O stripping do stress (remover dígitos 0–2 no fim, p.ex. `AE1→AE`) **não é feito aqui**;
            este mapeamento assume as chaves ARPAbet base (ex.: `AE`, `IY`, `CH`). O stripping ocorre
            na fase que chama este mapeamento.
          * Pausas/ruído `SIL`, `SPN`, `NSN` também possuem candidatos (`"sil"`, `"spn"`, `"nsn"`).
            Se estiverem no vocabulário, serão mapeados; em etapas posteriores, esses tokens
            normalmente são filtrados como não-fonêmicos.
          * Quando existem múltiplos candidatos (p.ex. `ER→["ɝ","ɚ","er"]`), a **ordem dos candidatos**
            define a preferência. Assim, se o vocabulário tiver `ɝ`, mapeia para `ɝ`; caso não, tenta
            `ɚ`; caso não, `"er"`.
          * O mapeamento é **determinístico** dado o `model_vocab` e a lista de candidatos embutida.

        Args:
          model_vocab (Dict[str, int]):
              Vocabulário do modelo (tipicamente `processor.tokenizer.get_vocab()`), mapeando
              `token -> id`. As chaves devem ser strings exatamente iguais às usadas pelo
              tokenizador do modelo (ex.: `{"æ": 123, "tʃ": 87, "ɹ": 45, "sil": 0, ...}`).

        Returns:
          Dict[str, str]:
              Dicionário `arpa -> token_modelo`. Inclui apenas as entradas ARPAbet cujos
              candidatos aparecem no `model_vocab`. Entradas sem nenhum candidato presente
              são omitidas.

        Examples:
          >>> vocab = {"æ": 10, "i": 11, "tʃ": 12, "sil": 0, "ɹ": 13}
          >>> build_arpabet_to_model_vocab_map(vocab)
          {'AE': 'æ', 'IY': 'i', 'CH': 'tʃ', 'SIL': 'sil', 'R': 'ɹ'}

        Quando usar:
          * Antes de converter uma sequência ARPAbet (vinda do G2P) para os símbolos aceitos
            pelo modelo de CTC (IPA/TIMIT).
          * Em pipelines que precisam de compatibilidade entre G2P (ARPAbet) e o inventário
            fonético do modelo.

        Quando não usar:
          * Se seu G2P já produz diretamente os símbolos do vocabulário do modelo.
          * Se o modelo usa um inventário incompatível (por ex., outra convenção que não IPA/TIMIT).

        Boas práticas:
          * Verifique se os tokens do seu modelo estão em **lowercase/uppercase** compatível
            com os candidatos. Os candidatos desta função já estão alinhados aos modelos
            baseados em TIMIT/IPA comuns em ASR/CTC.
          * Caso use um modelo com inventário diferente, adapte a lista de candidatos
            (constante interna) para refletir suas preferências.
    """
    present = set(model_vocab.keys())
    out: Dict[str, str] = {}
    for arpa, cands in _ARPA_TO_IPA_CANDIDATES.items():
        for cand in cands:
            if cand in present:
                out[arpa] = cand
                break
    return out

from typing import List, Dict, Tuple

def map_ref_arpabet_to_model_symbols(ref_arpa_seq: List[str],
                                     arpa2sym: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Converte uma sequência ARPAbet (com stress opcional) para símbolos do modelo.

    Regras:
      - Remove stress (0–2) no fim do token ARPAbet.
      - Ignora pausas/ruído (SIL, SPN, NSN) completamente (não entra em mapped, nem em missing).
      - Se houver mapeamento em `arpa2sym` e o símbolo mapeado for não-fonêmico
        ("sil", "spn", "nsn"), também ignora.
      - Mantém a ordem original em `mapped`.
      - Retorna `missing` como lista única e ordenada (ordem alfabética).

    Args:
      ref_arpa_seq: lista de tokens ARPAbet possivelmente com stress (ex.: ["AE1","T","ER0","SIL"]).
      arpa2sym: mapeamento ARPAbet base (sem stress) -> símbolo do vocabulário do modelo.

    Returns:
      (mapped, missing)
        mapped: lista de símbolos do modelo (IPA/TIMIT) já sem pausas.
        missing: bases ARPAbet sem mapeamento, únicas e ordenadas (ex.: ["ER","F"]).
    """
    NON_PHONEMIC_BASES = {"SIL", "SPN", "NSN"}
    NON_PHONEMIC_SYMS  = {"sil", "spn", "nsn"}

    mapped: List[str] = []
    missing: List[str] = []

    for ph in ref_arpa_seq:
        base = strip_stress(ph)

        # Ignora completamente pausas/ruído na referência.
        if base in NON_PHONEMIC_BASES:
            continue

        sym = arpa2sym.get(base)
        if sym is None:
            missing.append(base)
            continue

        # Mesmo se houver mapeamento, se cair em símbolo não-fonêmico, descarta.
        if sym in NON_PHONEMIC_SYMS:
            continue

        mapped.append(sym)

    return mapped, sorted(set(missing))



# ========================= Correções e normalização =========================

NON_PHONEMIC = {"|", "<s>", "</s>", " ", "sil", "spn", "nsn", ""}

def filter_non_phonemic(seq: List[str]) -> List[str]:
    return [p for p in seq if p not in NON_PHONEMIC]

_DIPH_SPLITS = {
    "eɪ": ["e", "ɪ"],
    "oʊ": ["o", "ʊ"],
    "aɪ": ["a", "ɪ"],
    "aʊ": ["a", "ʊ"],
    "ɔɪ": ["ɔ", "ɪ"],
}

def gop_lr_excluding_self(seg_log_post: np.ndarray, li: int, rival_mask: np.ndarray) -> float:
    """
    Calcula GOP-LR para o fonema 'li' em um segmento de log-posteriors (seg_log_post: [L,V]),
    usando como rival o MAIOR log-p entre as classes indicadas por 'rival_mask',
    mas **excluindo a própria coluna 'li'** do máximo.

    Retorna média em frames de (log_p(li) - max_log_p(rivais_excluindo_li)).
    """
    if seg_log_post.size == 0 or li is None or li < 0:
        return float("-inf")
    if rival_mask is None or rival_mask.size == 0:
        return float("-inf")

    # Exclui o próprio 'li' do conjunto de rivais
    mask_excl = rival_mask.copy()
    if 0 <= li < mask_excl.size:
        mask_excl[li] = False
    if not mask_excl.any():
        return float("-inf")

    rival = np.max(seg_log_post[:, mask_excl], axis=1)  # [L]
    log_p_l = seg_log_post[:, li]                       # [L]
    return float(np.mean(log_p_l - rival))


def expand_diphthongs_if_missing(phones: List[str], model_vocab: Dict[str, int]) -> List[str]:
    """Expande ditongos *apenas quando o modelo não possui o token composto*.

        Muitos vocabulários de ASR/CTC trazem vogais como unidades simples (``"e"``,
        ``"ɪ"`` etc.) e **não** incluem o ditongo como um único token (p.ex. ``"eɪ"``).
        Esta função detecta ditongos conhecidos e os **divide em duas vogais** somente
        se:
          1) o ditongo **não** estiver em ``model_vocab``; **e**
          2) **ambas** as partes estiverem em ``model_vocab``.

        Caso contrário, o símbolo é mantido como veio.

        Ditongos suportados (lado esquerdo → partes):
          * ``"eɪ" → ["e", "ɪ"]``
          * ``"oʊ" → ["o", "ʊ"]``
          * ``"aɪ" → ["a", "ɪ"]``
          * ``"aʊ" → ["a", "ʊ"]``
          * ``"ɔɪ" → ["ɔ", "ɪ"]``

        Observações importantes:
          * **Não** re-divide símbolos que já existem no vocabulário do modelo. Ex. se
            ``"eɪ"`` existir em ``model_vocab``, ele é preservado.
          * **Não** cria símbolos novos: se alguma parte do ditongo não existir no
            vocabulário (ex.: modelo sem ``"ɔ"``), o ditongo composto é mantido.
          * A ordem e o comprimento da sequência podem mudar (quando há expansão),
            mas os demais símbolos são preservados.
          * A função **não** normaliza equivalências (p.ex. ``ʤ → dʒ``), **não**
            filtra pausas/ruído (``sil``, ``spn``, ``nsn``) e **não** verifica
            fonotática; ela só trata a expansão de ditongos listados acima.

        Args:
          phones (List[str]):
              Sequência de símbolos alvo já mapeados para o “alfabeto” do modelo
              (tipicamente após ``map_ref_arpabet_to_model_symbols``). Pode conter
              ditongos e monoftongos.
          model_vocab (Dict[str, int]):
              Vocabulário do modelo (``token → id``). Usado para checar se o
              ditongo e/ou suas partes existem no modelo.

        Returns:
          List[str]: Nova sequência de símbolos, onde ditongos foram divididos em duas
          vogais **somente** quando o token composto faltava no vocabulário mas as
          duas partes existiam.

        Exemplos:
          >>> vocab = {"k":0, "e":1, "ɪ":2, "t":3}            # sem "eɪ"
          >>> expand_diphthongs_if_missing(["k","eɪ","t"], vocab)
          ['k', 'e', 'ɪ', 't']

          >>> vocab = {"k":0, "eɪ":1, "t":2}                  # com "eɪ"
          >>> expand_diphthongs_if_missing(["k","eɪ","t"], vocab)
          ['k', 'eɪ', 't']

          >>> vocab = {"k":0, "ɪ":1, "t":2}                   # sem "e" ⇒ não divide
          >>> expand_diphthongs_if_missing(["k","eɪ","t"], vocab)
          ['k', 'eɪ', 't']

        Quando usar:
          * Após mapear ARPAbet → símbolos do modelo, para aumentar a cobertura em
            modelos que não possuem tokens de ditongo.

        Quando não usar:
          * Se o seu modelo já contém todos os ditongos relevantes como tokens únicos.
          * Se você precisa de outras normalizações/expansões (essas etapas são
            tratadas por outras funções, p.ex. ``normalize_phone_sequence``).

        Desempenho:
          * Complexidade O(N), com N = número de símbolos em ``phones``.
          * Não aloca estruturas grandes; seguro para uso em lote.
   """
    out: List[str] = []
    for ph in phones:
        if ph in model_vocab:
            out.append(ph)
        elif ph in _DIPH_SPLITS:
            parts = _DIPH_SPLITS[ph]
            if all((p in model_vocab) for p in parts):
                out.extend(parts)
            else:
                out.append(ph)
        else:
            out.append(ph)
    return out

_EQUIV_PHONES = {
    "ʧ": "tʃ",
    "ʤ": "dʒ",
    "ɡ": "g",
    "r": "ɹ",
    "ɚ": "ɝ",
    # "ɜ": "ə",
}

def normalize_phone_symbol(p: str) -> str:
    return _EQUIV_PHONES.get(p, p)

def normalize_phone_sequence(seq: List[str]) -> List[str]:
    return [normalize_phone_symbol(p) for p in seq]


# ========================= Levenshtein acelerado =========================

if _HAS_RAPIDFUZZ:
    def levenshtein_ops(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
        """Calcula, em nível de **tokens**, as contagens de operações de
            Levenshtein mínimas entre uma sequência de referência e uma hipótese.

            Retorna uma tupla ``(S, D, I, N_ref)`` onde:
              * ``S`` — número de **substituições** (replace);
              * ``D`` — número de **deleções** (delete) da referência;
              * ``I`` — número de **inserções** (insert) na hipótese;
              * ``N_ref`` — comprimento da sequência de referência (``len(ref)``).

            Características e convenções:
              - Métrica clássica de Levenshtein de custo unitário:
                substituição = deleção = inserção = 1.
              - **Não** é Damerau–Levenshtein: transposições não têm custo especial.
              - As contagens são computadas de modo a minimizar o custo total.
              - Os tokens são tratados como unidades atômicas (string exata, sensível a
                maiúsculas/minúsculas). Se precisar, normalize antes (p.ex. lowercasing).
              - A soma ``S + D + I`` é a distância de edição mínima.
              - Para o cálculo de PER, use: ``PER = (S + D + I) / max(1, N_ref)``.
              - Implementação:
                 * Se disponível, utiliza **RapidFuzz** para obter as edições (rápido).
                 * Caso contrário, usa DP em Python puro com mesma semântica.

            Complexidade:
              - Com RapidFuzz: otimizada (na prática, próxima de linear para muitos casos).
              - Fallback DP: O(len(ref) * len(hyp)) em tempo e O(len(ref) * len(hyp)) em memória.

            Parâmetros
            ----------
            ref : List[str]
                Sequência de referência (p.ex., fonemas alvo).
            hyp : List[str]
                Sequência hipotética/produzida (p.ex., fonemas reconhecidos).

            Returns
            -------
            Tuple[int, int, int, int]
                ``(S, D, I, N_ref)`` conforme descrito acima.

            Exemplos
            --------
            >>> levenshtein_ops(["a","b","c"], ["a","x","c","d"])
            (1, 0, 1, 3)   # b→x (S=1), inserção de d (I=1)

            >>> levenshtein_ops(["p","t"], ["p"])
            (0, 1, 0, 2)   # deleção de t

            >>> levenshtein_ops([], ["k","a"])
            (0, 0, 2, 0)   # duas inserções, N_ref = 0

            Boas práticas
            -------------
            * Pré-normalize os tokens (p.ex., mapeie símbolos equivalentes, remova pausas).
            * Para relatórios estratificados (por classe/posição), derive as máscaras a
              partir dos índices da referência e aplique sobre estas contagens.
        """
        ops = RFLev.editops(ref, hyp)  # trabalha diretamente em listas
        S = D = I = 0
        for op in ops:
            if op.tag == "replace": S += 1
            elif op.tag == "delete": D += 1
            else: I += 1
        return S, D, I, len(ref)

    def levenshtein_backtrace(ref: List[str], hyp: List[str]) -> List[Tuple[str, Optional[int], Optional[int]]]:
        """
        Reconstrói alinhamento com 'match' explícito antes de cada operação.
        """
        ops = RFLev.editops(ref, hyp)
        out: List[Tuple[str, Optional[int], Optional[int]]] = []
        i = j = 0
        for op in ops:
            # matches até o ponto da operação
            while i < op.src_pos and j < op.dest_pos:
                out.append(("match", i, j))
                i += 1; j += 1
            if op.tag == "replace":
                out.append(("sub", op.src_pos, op.dest_pos))
                i = op.src_pos + 1
                j = op.dest_pos + 1
            elif op.tag == "delete":
                out.append(("del", op.src_pos, None))
                i = op.src_pos + 1
            else:  # insert
                out.append(("ins", None, op.dest_pos))
                j = op.dest_pos + 1
        # matches finais
        while i < len(ref) and j < len(hyp):
            out.append(("match", i, j))
            i += 1; j += 1
        return out
else:
    def levenshtein_ops(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
        n, m = len(ref), len(hyp)
        dp = [[(0,0,0,0) for _ in range(m+1)] for _ in range(n+1)]
        for i in range(1, n+1):
            c,S,D,I = dp[i-1][0]
            dp[i][0] = (c+1, S, D+1, I)
        for j in range(1, m+1):
            c,S,D,I = dp[0][j-1]
            dp[0][j] = (c+1, S, D, I+1)
        for i in range(1, n+1):
            ri = ref[i-1]
            for j in range(1, m+1):
                hj = hyp[j-1]
                c0,S0,D0,I0 = dp[i-1][j-1]
                cand1 = (c0, S0, D0, I0) if ri==hj else (c0+1, S0+1, D0, I0)
                c1,S1,D1,I1 = dp[i-1][j]
                cand2 = (c1+1, S1, D1+1, I1)
                c2,S2,D2,I2 = dp[i][j-1]
                cand3 = (c2+1, S2, D2, I2+1)
                dp[i][j] = min([cand1, cand2, cand3], key=lambda x: x[0])
        _,S,D,I = dp[n][m][0],dp[n][m][1],dp[n][m][2],dp[n][m][3]
        return S, D, I, n

    def levenshtein_backtrace(ref: List[str], hyp: List[str]) -> List[Tuple[str, Optional[int], Optional[int]]]:
        n, m = len(ref), len(hyp)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(1, n+1): dp[i][0] = i
        for j in range(1, m+1): dp[0][j] = j
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0 if ref[i-1]==hyp[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost,
                )
        i, j = n, m
        ops = []
        while i > 0 or j > 0:
            if i>0 and j>0 and dp[i][j] == dp[i-1][j-1] + (0 if ref[i-1]==hyp[j-1] else 1):
                if ref[i-1]==hyp[j-1]:
                    ops.append(("match", i-1, j-1))
                else:
                    ops.append(("sub", i-1, j-1))
                i -= 1; j -= 1
            elif i>0 and dp[i][j] == dp[i-1][j] + 1:
                ops.append(("del", i-1, None)); i -= 1
            else:
                ops.append(("ins", None, j-1)); j -= 1
        ops.reverse()
        return ops


def confusion_list(ref_w: List[str], hyp_w: List[str]) -> List[Dict[str, Optional[str]]]:
    ops = levenshtein_backtrace(ref_w, hyp_w)
    out = []
    for kind, i_ref, j_hyp in ops:
        out.append({
            "op": kind,
            "ref": (ref_w[i_ref] if i_ref is not None and i_ref < len(ref_w) else None),
            "hyp": (hyp_w[j_hyp] if j_hyp is not None and j_hyp < len(hyp_w) else None),
        })
    return out

def summarize_ops(ref_w: List[str], hyp_w: List[str]) -> List[str]:
    ops = levenshtein_backtrace(ref_w, hyp_w)
    out = []
    for kind, i_ref, j_hyp in ops:
        if kind == "sub":
            out.append(f"sub({ref_w[i_ref]}→{hyp_w[j_hyp]})")
        elif kind == "del":
            out.append(f"del({ref_w[i_ref]})")
        elif kind == "ins":
            out.append(f"ins({hyp_w[j_hyp]})")
        else:
            out.append("match")
    return out


# ========================= CTC helpers =========================

def ctc_collapse(ids: List[int], blank_id: int) -> List[int]:
    out, prev = [], None
    for i in ids:
        if i != blank_id and i != prev:
            out.append(i)
        prev = i
    return out

@dataclass
class PhoneSegment:
    phone_id: int
    t1: int
    t2: int
    start_s: float
    end_s: float

def build_ctc_targets(phone_ids: List[int], blank_id: int) -> List[int]:
    out = [blank_id]
    for pid in phone_ids:
        out.extend([pid, blank_id])
    return out

def ctc_force_align(log_post: np.ndarray, ref_ids: List[int], blank_id: int) -> List[PhoneSegment]:
    """Alinha, via **Viterbi/CTC em log-domínio**, uma sequência de rótulos alvo
        (fonemas) aos quadros de probabilidade do modelo e retorna segmentos por
        fonema com limites de quadro.

        A função implementa um alinhamento forçado simples para CTC:
        1) Constrói a sequência-alvo **estendida** ``y'`` intercalando BLANKs
           (ver ``build_ctc_targets``): ``[blank, p1, blank, p2, ..., pN, blank]``.
        2) Executa *dynamic programming* (treliça T×M, em log-prob) com duas
           transições por passo temporal: **stay** (permanece no mesmo estado j) ou
           **advance** (avança para j+1). O custo de cada célula é a soma do melhor
           custo anterior com o log-p do rótulo de ``y'[j]`` no quadro t.
        3) Faz *backtrace* do último estado para o início e, para cada estado
           de **rótulo** (índices ímpares em ``y'``), extrai o menor e o maior
           índice de quadro visitado, produzindo (t1, t2) **inclusivos**.

        Observações importantes
        -----------------------
        - A entrada deve ser ``log_post`` (log-softmax dos logits). Se você passar
          logits crus, o alinhamento ficará enviesado.
        - O alinhamento é do tipo **Viterbi** (caminho máximo). Não calcula
          pós-termo esperado (*forward–backward*).
        - Os campos ``start_s`` e ``end_s`` dos segmentos são retornados como
          ``-1.0`` e devem ser preenchidos pelo chamador, mapeando quadro→tempo
          (p.ex., com uma função do tipo ``_frame_to_time``).
        - Se um estado de rótulo não for visitado no caminho ótimo, seu par
          (t1, t2) será ``(-1, -1)``.
        - A função **não** impõe duração mínima por fonema; isso é responsabilidade
          do chamador (veja ``_enforce_min_frames`` no código).

        Parâmetros
        ----------
        log_post : np.ndarray
            Matriz ``[T, V]`` de log-probabilidades por quadro (``log_softmax``),
            onde ``T`` é o número de quadros de áudio e ``V`` é o tamanho do
            vocabulário do modelo CTC.
        ref_ids : List[int]
            Sequência de IDs de fonemas (sem BLANK) correspondente ao alvo a ser
            alinhado. **A ordem importa**.
        blank_id : int
            ID do token BLANK no vocabulário (comum ser o ``pad_token_id``).

        Returns
        -------
        List[PhoneSegment]
            Lista com um ``PhoneSegment`` por **fonema de referência** (mesma
            cardinalidade de ``ref_ids``), com:
            - ``phone_id``: ID do fonema (de ``ref_ids``),
            - ``t1`` / ``t2``: índices de quadro **inclusivos** do segmento,
              ou ``-1`` se não alinhado,
            - ``start_s`` / ``end_s``: `-1.0` (o chamador deve converter quadros
              em segundos conforme a duração/``T``).

        Erros / casos de borda
        -----------------------
        - ``ref_ids`` vazio → retorna lista vazia.
        - Dimensões inválidas de ``log_post`` (p.ex. ``V`` não contém ``blank_id``)
          resultarão em índices inválidos na treliça.
        - Se o áudio for muito curto (T pequeno), alguns fonemas podem ficar com
          ``(-1, -1)`` (sem cobertura).

        Complexidade
        ------------
        - Tempo: :math:`O(T · M)`, onde :math:`M = 2·len(ref_ids)+1` (sequência
          estendida com BLANKs).
        - Espaço: :math:`O(T · M)` para a treliça e *backpointers*.

        Quando usar
        -----------
        - Para obter **delimitações de quadro** por fonema em um alvo conhecido,
          a partir de saídas CTC (p.ex., para calcular GOP por segmento).
        - Para pós-processar decodificações e extrair *timing* aproximado.

        Quando **não** usar
        -------------------
        - Se você precisa de **médias** ou **incerteza** por estado, use um
          alinhamento probabilístico (forward–backward).
        - Se não estiver trabalhando com CTC/log-post, mas com ASR de atenção
          *end-to-end* (use alinhamento de atenção/CTC-seg).

        Exemplos
        --------
        >>> # log_post: (T=160, V=50), ref com 3 fonemas
        >>> segs = ctc_force_align(log_post, ref_ids=[12, 7, 33], blank_id=0)
        >>> for s in segs:
        ...     print(s.phone_id, s.t1, s.t2)
        12  10  18
        7   19  31
        33  32  45

        Dica
        ----
        Após obter os segmentos, aplique uma política de duração mínima, p.ex.:
        ``t1, t2 = _enforce_min_frames(t1, t2, T, min_frames=2)``, antes de
        medir métricas por fonema (GOP, etc.).
    """
    T, V = log_post.shape
    target = build_ctc_targets(ref_ids, blank_id)
    M = len(target)

    trellis = np.full((T, M), -np.inf, dtype=np.float32)
    backp   = np.zeros((T, M), dtype=np.int8)
    trellis[0, 0] = log_post[0, target[0]]

    for t in range(1, T):
        stay = trellis[t-1, :] + log_post[t, target]
        adv  = np.full(M, -np.inf, dtype=np.float32)
        adv[1:] = trellis[t-1, :-1] + log_post[t, target[1:]]
        better_adv = adv > stay
        trellis[t] = np.where(better_adv, adv, stay)
        backp[t]   = np.where(better_adv, 1, 0)

    j = M - 1
    t = T - 1
    path = []
    while t >= 0 and j >= 0:
        path.append((t, j))
        if backp[t, j] == 1:
            j -= 1
        t -= 1
    path.reverse()

    tmin = [None]*M
    tmax = [None]*M
    for (t, j) in path:
        if tmin[j] is None:
            tmin[j] = t
        tmax[j] = t

    segs: List[PhoneSegment] = []
    for idx, pid in enumerate(ref_ids):
        j_phone = 2*idx + 1
        if j_phone < M and tmin[j_phone] is not None and tmax[j_phone] is not None:
            t1, t2 = int(tmin[j_phone]), int(tmax[j_phone])
        else:
            t1, t2 = -1, -1
        segs.append(PhoneSegment(phone_id=pid, t1=t1, t2=t2, start_s=-1.0, end_s=-1.0))
    return segs


# ========================= GOP =========================

def _enforce_min_frames(t1: int, t2: int, T: int, min_frames: int = 2) -> Tuple[int, int]:
    """Garante uma **duração mínima** (em quadros) para um segmento [t1, t2] inclusivo.

        A função tenta expandir simetricamente o segmento para possuir pelo menos
        ``min_frames`` quadros, respeitando os limites do sinal (0 … T-1). Caso o
        intervalo de entrada seja inválido (índices negativos, ``t2 < t1``) o par
        é retornado **inalterado**. Se não houver espaço suficiente para atingir
        a duração mínima, retorna a maior expansão possível dentro de [0, T-1].

        Uso típico: pós-processar limites vindos de um alinhamento CTC (p.ex.
        ``ctc_force_align``) antes de calcular métricas que precisam de alguns
        quadros por fonema (GOP, média de *log-post*, etc.).

        Parâmetros
        ----------
        t1 : int
            Quadro inicial **inclusivo** do segmento. Pode ser -1 para “sem alinhamento”.
        t2 : int
            Quadro final **inclusivo** do segmento. Pode ser -1 para “sem alinhamento”.
        T : int
            Número total de quadros do sinal (dimensão temporal de ``log_post``).
        min_frames : int, opcional (padrão=2)
            Duração mínima desejada em quadros. Deve ser >= 1.

        Retorna
        -------
        (int, int)
            Par ``(t1_adj, t2_adj)`` ajustado. Se o intervalo original for inválido,
            retorna ``(t1, t2)`` sem alterações. Caso válido, o intervalo ajustado:
            - tem ao menos ``min_frames`` quadros, se possível;
            - fica recortado a [0, T-1] quando bater nos limites.

        Notas
        -----
        - Expansão é feita **aproximadamente simétrica**: a metade esquerda é
          ``need // 2`` e a direita é ``need - left``; um pequeno laço final tenta
          compensar sobras até atingir ``min_frames`` (ou os limites).
        - Complexidade: O(1).
        - Se ``t1`` ou ``t2`` forem negativos, **nenhum ajuste** é aplicado.

        Exemplos
        --------
        >>> _enforce_min_frames(10, 10, T=100, min_frames=3)
        (9, 11)         # expandiu 1 à esquerda e 1 à direita

        >>> _enforce_min_frames(0, 0, T=2, min_frames=4)
        (0, 1)          # não há quadros suficientes; usa a maior janela possível

        >>> _enforce_min_frames(-1, -1, T=100, min_frames=3)
        (-1, -1)        # intervalo inválido permanece inalterado

        Quando usar
        -----------
        - Antes de computar GOP ou médias de log-prob por segmento, evitando
          instabilidade em trechos muito curtos.

        Quando **não** usar
        -------------------
        - Se você deseja **invalidar** segmentos curtos (em vez de expandi-los);
          neste caso, trate ``(t1, t2)`` externamente e descarte-os.
    """
    if t1 < 0 or t2 < 0 or t2 < t1:
        return t1, t2
    length = t2 - t1 + 1
    if length >= min_frames:
        return t1, t2
    need = min_frames - length
    left = need // 2
    right = need - left
    t1_new = max(0, t1 - left)
    t2_new = min(T - 1, t2 + right)
    while (t2_new - t1_new + 1) < min_frames:
        if t1_new > 0: t1_new -= 1
        elif t2_new < T - 1: t2_new += 1
        else: break
    return t1_new, t2_new

def gop_lr_with_mask_premax(seg_log_post: np.ndarray,
                            li: int,
                            rival_max_all: Optional[np.ndarray],
                            rival_max_vowel: Optional[np.ndarray],
                            rival_max_cons: Optional[np.ndarray],
                            cls_is_vowel: Optional[bool]) -> float:
    """Calcula o GOP (log-ratio) de um segmento usando rivais já **pré-maximizados**.

    A métrica retorna a média, ao longo dos quadros do segmento, de
    ``log P(phone=li) − max_rival log P(rival)``. O conjunto de rivais
    pode ser:
      - **vogais** (``rival_max_vowel``) quando ``cls_is_vowel is True``;
      - **consoantes** (``rival_max_cons``) quando ``cls_is_vowel is False``;
      - **todos** (``rival_max_all``) quando ``cls_is_vowel is None`` ou quando
        o vetor específico estiver indisponível.

    Os vetores ``rival_max_*`` devem ter sido **pré-computados por quadro**,
    tomando o máximo nas colunas de ``seg_log_post`` correspondentes ao conjunto
    de rivais escolhido (ex.: aplicando uma máscara booleana e fazendo
    ``np.max(seg_lp[:, mask], axis=1)``). Essa estratégia evita custo repetido
    de *reductions* por fonema e acelera o cálculo em lote.

    Parâmetros
    ----------
    seg_log_post : np.ndarray, shape (L, V)
        Log-posteriors (tipicamente ``log_softmax``) do modelo para o
        **segmento temporal do fonema alvo**; L = nº de quadros do segmento,
        V = tamanho do vocabulário de símbolos do modelo.
    li : int
        Índice do símbolo alvo (coluna em ``seg_log_post``) para o qual o GOP
        será calculado.
    rival_max_all : Optional[np.ndarray], shape (L,)
        Para cada quadro, o **máximo** de log-posteriors dentre todos os rivais
        válidos (excluindo *blank* e tokens não-fonêmicos).
    rival_max_vowel : Optional[np.ndarray], shape (L,)
        Máximo por quadro **apenas** entre rivais do conjunto de **vogais**.
    rival_max_cons : Optional[np.ndarray], shape (L,)
        Máximo por quadro **apenas** entre rivais do conjunto de **consoantes**.
    cls_is_vowel : Optional[bool]
        Classe do fonema alvo: ``True`` = vogal, ``False`` = consoante,
        ``None`` = desconhecida (usa ``rival_max_all``).

    Retorna
    -------
    float
        GOP-LR do segmento: média de ``log_p_l − log_p_rival_max`` ao longo dos
        L quadros. Retorna ``-inf`` quando:
        - ``seg_log_post`` é vazio,
        - não há vetor de rivais apropriado (``None`` ou vazio).

    Fórmula
    -------
    Para cada quadro *t* (0 … L−1):
        ``g_t = log_post[t, li] − rival_max[t]``
    GOP do segmento:
        ``GOP = mean_t(g_t)``

    Comportamento de borda
    ----------------------
    - Se o segmento não tiver quadros (``L == 0``), retorna ``-inf``.
    - Se o vetor de rivais selecionado estiver ausente ou vazio, retorna ``-inf``.
    - Se ``li`` estiver fora de [0, V−1], ocorrerá *IndexError*; deve ser
      garantido pelo chamador que o índice do alvo é válido.

    Requisitos de entrada
    ---------------------
    - ``seg_log_post`` deve estar no **mesmo espaço** (log-probabilidades) que
      os vetores ``rival_max_*`` foram obtidos (i.e., todos em *log-space*).
    - Os vetores ``rival_max_*`` devem ter **shape (L,)** e corresponder ao
      mesmo recorte temporal (mesmo L) do segmento.

    Exemplos
    --------
    >>> # seg_log_post: (L=3, V=5), alvo li=2
    >>> # rivais pré-max: all
    >>> gop = gop_lr_with_mask_premax(seg_lp, li=2,
    ...                                rival_max_all=np.array([-1.2, -0.8, -1.5]),
    ...                                rival_max_vowel=None,
    ...                                rival_max_cons=None,
    ...                                cls_is_vowel=None)
    >>> float(gop)
    0.37  # exemplo ilustrativo

    Quando usar
    -----------
    - Em pipelines de GOP em **lote**, após pré-computar os máximos por conjunto
      (todos/vogais/consoantes) para cada segmento, reduzindo custo por fonema.
    - Quando deseja aplicar *thresholds* por fonema ou por classe (vogal/cons.)
      na etapa seguinte.

    Quando **não** usar
    -------------------
    - Se você **não** possui os rivais pré-maximizados; nesse caso, use uma
      variante que aceite a máscara e faça o ``np.max`` internamente (mais
      simples, porém mais lenta).
    - Quando o segmento não possui quadros válidos; convém tratar o caso
      previamente (p.ex., marcando o fonema como ausente).
    """
    if seg_log_post.size == 0:
        return float("-inf")
    if cls_is_vowel is True and rival_max_vowel is not None and rival_max_vowel.size:
        rival = rival_max_vowel
    elif cls_is_vowel is False and rival_max_cons is not None and rival_max_cons.size:
        rival = rival_max_cons
    else:
        rival = rival_max_all
    if rival is None or rival.size == 0:
        return float("-inf")
    log_p_l = seg_log_post[:, li]
    return float(np.mean(log_p_l - rival))


# ========================= Traços articulatórios (AER) =========================

FEATURE_WEIGHTS = {
    "voi": 1.5, "man": 1.0, "pla": 1.0, "lat": 0.5, "asp": 0.5,  # consoantes
    "h": 1.0, "b": 1.0, "round": 1.2, "rh": 0.5,                 # vogais
}

_FEATURES = {
    # CONSOANTES
    "p":{"voi":0,"man":"stop","pla":"bilab","lat":0,"asp":0},
    "b":{"voi":1,"man":"stop","pla":"bilab","lat":0,"asp":0},
    "t":{"voi":0,"man":"stop","pla":"alveo","lat":0,"asp":0},
    "d":{"voi":1,"man":"stop","pla":"alveo","lat":0,"asp":0},
    "k":{"voi":0,"man":"stop","pla":"velar","lat":0,"asp":0},
    "g":{"voi":1,"man":"stop","pla":"velar","lat":0,"asp":0}, "ɡ":{"voi":1,"man":"stop","pla":"velar","lat":0,"asp":0},
    "tʃ":{"voi":0,"man":"affr","pla":"post","lat":0,"asp":0},
    "ʤ":{"voi":1,"man":"affr","pla":"post","lat":0,"asp":0}, "dʒ":{"voi":1,"man":"affr","pla":"post","lat":0,"asp":0},
    "f":{"voi":0,"man":"fric","pla":"labdent","lat":0,"asp":0},
    "v":{"voi":1,"man":"fric","pla":"labdent","lat":0,"asp":0},
    "θ":{"voi":0,"man":"fric","pla":"dent","lat":0,"asp":0},
    "ð":{"voi":1,"man":"fric","pla":"dent","lat":0,"asp":0},
    "s":{"voi":0,"man":"fric","pla":"alveo","lat":0,"asp":0},
    "z":{"voi":1,"man":"fric","pla":"alveo","lat":0,"asp":0},
    "ʃ":{"voi":0,"man":"fric","pla":"post","lat":0,"asp":0},
    "ʒ":{"voi":1,"man":"fric","pla":"post","lat":0,"asp":0},
    "h":{"voi":0,"man":"fric","pla":"glott","lat":0,"asp":1},
    "m":{"voi":1,"man":"nasal","pla":"bilab","lat":0,"asp":0},
    "n":{"voi":1,"man":"nasal","pla":"alveo","lat":0,"asp":0},
    "ŋ":{"voi":1,"man":"nasal","pla":"velar","lat":0,"asp":0},
    "l":{"voi":1,"man":"lat","pla":"alveo","lat":1,"asp":0},
    "ɹ":{"voi":1,"man":"appr","pla":"post","lat":0,"asp":0},
    "w":{"voi":1,"man":"appr","pla":"labvel","lat":0,"asp":0},
    "j":{"voi":1,"man":"appr","pla":"pal","lat":0,"asp":0},

    # VOGAIS
    "i":{"h":"close","b":"front","round":0,"rh":0},
    "ɪ":{"h":"close","b":"front","round":0,"rh":0},
    "eɪ":{"h":"close-mid","b":"front","round":0,"rh":0},
    "ɛ":{"h":"open-mid","b":"front","round":0,"rh":0},
    "æ":{"h":"open","b":"front","round":0,"rh":0},
    "ɑ":{"h":"open","b":"back","round":0,"rh":0},
    "ʌ":{"h":"open-mid","b":"central","round":0,"rh":0},
    "ɝ":{"h":"mid","b":"central","round":0,"rh":1},
    "ə":{"h":"mid","b":"central","round":0,"rh":0},
    "oʊ":{"h":"close-mid","b":"back","round":1,"rh":0},
    "ɔ":{"h":"open-mid","b":"back","round":1,"rh":0},
    "ʊ":{"h":"close","b":"back","round":1,"rh":0},
    "u":{"h":"close","b":"back","round":1,"rh":0},
    "aɪ":{"h":"open","b":"front","round":0,"rh":0},
    "aʊ":{"h":"open","b":"back","round":1,"rh":0},
    "ɔɪ":{"h":"open-mid","b":"back","round":1,"rh":0},
}

_CANON_FOR_FEATURES = {
    "iy":"i","ih":"ɪ","eh":"ɛ","ae":"æ","aa":"ɑ","ah":"ʌ","er":"ɝ","ax":"ə",
    "uw":"u","uh":"ʊ","ao":"ɔ","ow":"oʊ","aw":"aʊ","ay":"aɪ","oy":"ɔɪ",
    "jh":"dʒ","ch":"tʃ","hh":"h","sh":"ʃ","zh":"ʒ","ng":"ŋ","r":"ɹ","y":"j","g":"g","ɡ":"g",
}

@lru_cache(maxsize=None)
def _feature_weight(k: str) -> float:
    return float(FEATURE_WEIGHTS.get(k, 1.0))

@lru_cache(maxsize=10_000)
def _feat_vec_cached(ph: str):
    ph = _CANON_FOR_FEATURES.get(ph, ph)
    f = _FEATURES.get(ph)
    if not f: return {}, []
    return f, tuple(f.keys())

@lru_cache(maxsize=10_000)
def is_vowel_cached(ph: str) -> Optional[bool]:
    ph = _CANON_FOR_FEATURES.get(ph, ph)
    F = _FEATURES.get(ph)
    if not F:
        return None
    return any(k in F for k in ("h","b","round","rh"))

def is_vowel(ph: str) -> Optional[bool]:
    return is_vowel_cached(ph)

def _aer_expand_parts(ph: str) -> List[str]:
    parts = _DIPH_SPLITS.get(ph)
    if not parts:
        return [ph]
    ok = all((_CANON_FOR_FEATURES.get(p, p) in _FEATURES) for p in parts)
    return parts if ok else [ph]

def articulatory_distance_weighted_lists(ref_parts: List[str], hyp_parts: List[str]) -> Tuple[float,float,Dict[str,Tuple[float,float]],Tuple[float,float]]:
    """Calcula a distância articulatória ponderada entre duas listas de segmentos fonéticos.

        A função compara, posição a posição, os traços articulatórios de cada par
        (ref_parts[i], hyp_parts[i]) e acumula um **peso de desajuste** por traço.
        Traços suportados incluem (consoantes) *voi, man, pla, lat, asp* e (vogais)
        *h, b, round, rh*. Cada traço possui um peso (ver `FEATURE_WEIGHTS`), e o
        **AER** (Articulatory Error Rate) pode ser obtido como `mism_w / total_w`.

        Observações importantes
        -----------------------
        - Esta função **não** faz canonização nem divisão de ditongos; espera-se
          que `ref_parts` e `hyp_parts` já venham preparados (ex.: via
          `_aer_expand_parts` e normalizações anteriores).
        - Quando o comprimento difere, o excedente em `ref_parts` é tratado como
          **deleção** (conta 100% de desajuste para os traços do segmento faltante),
          e o excedente em `hyp_parts` como **inserção** (idem).
        - Segmentos sem entrada na tabela de traços (`_FEATURES`) simplesmente
          **não contribuem** para o total.

        Parâmetros
        ----------
        ref_parts : List[str]
            Lista de segmentos/partes de fonemas de referência (p.ex., fonemas
            simples ou componentes de ditongo já expandidos).
        hyp_parts : List[str]
            Lista de segmentos/partes produzidas/hipotéticas a serem comparadas
            à referência (mesma granularidade de `ref_parts`).

        Retorna
        -------
        Tuple[float, float, Dict[str, Tuple[float, float]], Tuple[float, float]]
            - mism_w : float
                Soma dos pesos de **desajuste** (mismatch) em todos os traços,
                agregada sobre todas as posições comparadas.
            - total_w : float
                Soma dos **pesos considerados** (denominador). O AER global é
                `mism_w / total_w` quando `total_w > 0`.
            - per_feat : Dict[str, Tuple[float, float]]
                Quebra por traço: `per_feat[feat] = (mism, total)` onde
                `mism/total` é o AER daquele traço específico.
            - legacy : Tuple[float, float]
                Par reservado para compatibilidade retroativa (não utilizado aqui):
                `(0.0, 0.0)`.

        Traços e pesos
        --------------
        - Consoantes: ``voi`` (vozeamento), ``man`` (modo), ``pla`` (ponto),
          ``lat`` (lateralidade), ``asp`` (aspiração).
        - Vogais: ``h`` (altura), ``b`` (anterioridade/backness), ``round``
          (arredondamento), ``rh`` (rótico).
        - Pesos definidos em `FEATURE_WEIGHTS` (p.ex., `voi: 1.5`, `round: 1.2`, etc.).

        Algoritmo (resumo)
        ------------------
        1. Para `i` de `0` a `min(len(ref_parts), len(hyp_parts)) - 1`:
           - Obtém os mapas de traços de `ref_parts[i]` e `hyp_parts[i]`.
           - Para cada traço em comum, soma `w` a `total_w` e, se os valores
             diferirem, soma `w` a `mism_w` (e ao `mism` do traço em `per_feat`).
        2. Para sobras em `ref_parts` (deleções) e em `hyp_parts` (inserções):
           - Para cada traço do segmento extra, soma `w` tanto a `total_w` quanto
             a `mism_w` (desajuste total), e atualiza `per_feat`.
        3. Retorna os acumulados.

        Casos de borda
        --------------
        - Se nenhum segmento tiver traços válidos, `total_w == 0` (o chamador deve
          tratar a divisão por zero ao calcular AER).
        - Segmentos desconhecidos (não mapeados em `_FEATURES`) são ignorados.
        - Listas com comprimentos diferentes geram penalidade total para os
          segmentos excedentes.

        Exemplo
        -------
        >>> ref_parts = ["t", "ʃ"]    # traços: stop/alveolar vs affricate/post
        >>> hyp_parts = ["t", "s"]    # segunda posição difere em modo/ponto
        >>> mism_w, total_w, per_feat, _ = articulatory_distance_weighted_lists(
        ...     ref_parts, hyp_parts
        ... )
        >>> aer_global = mism_w / total_w if total_w > 0 else 0.0
        >>> round(aer_global, 3)
        0.2  # valor ilustrativo; depende dos pesos em FEATURE_WEIGHTS
    """
    mism_w = 0.0
    total_w = 0.0
    per_feat: Dict[str, Tuple[float,float]] = {}
    L1, L2 = len(ref_parts), len(hyp_parts)
    L = min(L1, L2)

    for i in range(L):
        rph, hph = ref_parts[i], hyp_parts[i]
        F1, keys1 = _feat_vec_cached(rph)
        F2, keys2 = _feat_vec_cached(hph)
        keys = [k for k in keys1 if k in keys2]
        for k in keys:
            w = _feature_weight(k)
            total_w += w
            mw, tw = per_feat.get(k, (0.0, 0.0))
            tw += w
            if F1[k] != F2[k]:
                mism_w += w
                mw += w
            per_feat[k] = (mw, tw)

    # deleções
    for i in range(L, L1):
        rph = ref_parts[i]
        F1, keys1 = _feat_vec_cached(rph)
        for k in keys1:
            w = _feature_weight(k)
            total_w += w; mism_w += w
            mw, tw = per_feat.get(k, (0.0,0.0))
            per_feat[k] = (mw + w, tw + w)

    # inserções
    for i in range(L, L2):
        hph = hyp_parts[i]
        F2, keys2 = _feat_vec_cached(hph)
        for k in keys2:
            w = _feature_weight(k)
            total_w += w; mism_w += w
            mw, tw = per_feat.get(k, (0.0,0.0))
            per_feat[k] = (mw + w, tw + w)

    return mism_w, total_w, per_feat, (0.0, 0.0)


# ========================= Fluência (VAD leve) =========================

def _approx_frame_energy_and_zcr(audio: np.ndarray, T: int, sr: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
    """Estima, de forma leve, a energia por quadro e a taxa de cruzamentos por zero (ZCR).

        A rotina divide o sinal `audio` em **T** segmentos contíguos (sem sobreposição),
        igualmente espaçados em número de amostras, e computa para cada segmento:
        - **Energia aproximada**: média do valor absoluto (AM, *average magnitude*), mais
          barata que RMS e adequada para VAD rápido.
        - **ZCR** (*zero-crossing rate*): fração de pares de amostras consecutivos que
          cruzam o zero (mudança de sinal), normalizada para o intervalo [0, 1].

        Essa estimativa foi pensada para casar com o eixo temporal do modelo CTC:
        em geral, escolha `T` igual ao número de *frames* (p. ex. `logits.shape[0]`).

        Parâmetros
        ----------
        audio : np.ndarray
            Sinal de áudio mono (float/np.float32 recomendado), amostrado a `sr`.
        T : int
            Número de quadros/partições a produzir. Deve ser não negativo.
            Se `T == 0`, retorna dois vetores vazios.
        sr : int, opcional (default=16000)
            Taxa de amostragem em Hz. Não é usada diretamente no cálculo,
            mas mantida por simetria de API e para conversões externas de tempo.

        Retorna
        -------
        (am, zcr) : Tuple[np.ndarray, np.ndarray]
            - `am`  (shape: (T,), dtype float32): média do valor absoluto por quadro.
              Um pequeno `+1e-12` é somado para evitar zeros exatos.
            - `zcr` (shape: (T,), dtype float32): taxa de cruzamentos por zero por quadro,
              no intervalo [0, 1]. Para quadros de 1 amostra, a ZCR é 0.

        Detalhes de implementação
        -------------------------
        - A segmentação usa `np.linspace(0, n, T+1, dtype=int)` para gerar os *edges*,
          o que pode resultar em alguns quadros vazios (quando `T > n`); tais quadros
          retornam AM=0 e ZCR=0.
        - O cálculo percorre os quadros com um laço Python, mas cada amostra do sinal
          é lida no máximo uma vez; complexidade ≈ O(n + T).
        - O casting para `np.float32` é feito por quadro para manter consistência
          numérica e reduzir custo.

        Casos de borda
        --------------
        - `len(audio) == 0` ou `T <= 0` → retorna vetores de zeros com comprimento `T`.
        - Quadros vazios (sem amostras) → AM=0, ZCR=0.
        - Quadros com apenas 1 amostra → ZCR=0 (não há pares consecutivos).
        - Se o sinal contiver valores NaN/Inf, o resultado pode ser indefinido.

        Quando usar
        -----------
        - Como *features* simples para um VAD leve, *gating* de pausas e heurísticas
          de fluência.
        - Para obter máscaras de fala/pause rápidas em sincronia com `T` *frames* de
          uma rede acústica.

        Quando NÃO usar
        ---------------
        - Se precisar de medidas de energia mais robustas (RMS, log-energia com
          janelamento e *pre-emphasis*) ou ZCR com janelas sobrepostas e suavização.
          Nesse caso, prefira um *frontend* de *feature extraction* dedicado.

        Exemplo
        -------
        >>> am, z = _approx_frame_energy_and_zcr(audio, T=100, sr=16000)
        >>> am.shape, z.shape
        ((100,), (100,))
        >>> float(am.mean()) > 0, 0.0 <= float(z.max()) <= 1.0
        (True, True)
    """
    n = len(audio)
    if n == 0 or T <= 0:
        return np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
    edges = np.linspace(0, n, T+1, dtype=int)
    # energia por média do valor absoluto (AM) — mais leve que RMS
    am = np.zeros(T, dtype=np.float32)
    zcr = np.zeros(T, dtype=np.float32)
    for t in range(T):
        seg = audio[edges[t]:edges[t+1]]
        if len(seg) == 0:
            am[t] = 0.0; zcr[t] = 0.0; continue
        segf = seg.astype(np.float32)
        am[t] = float(np.mean(np.abs(segf)) + 1e-12)
        zcr[t] = float(((segf[:-1] * segf[1:]) < 0).mean() if len(segf) > 1 else 0.0)
    return am, zcr

# def _vad_mask_from_energy(am: np.ndarray, zcr: np.ndarray) -> np.ndarray:
#     if am.size == 0:
#         return np.array([], dtype=bool)
#     # thr_am = max(1e-8, float(np.percentile(am, 60)))
#     # thr_zc = float(np.percentile(zcr, 40))
#     thr_am = max(1e-8, float(np.percentile(am, 60)))
#     thr_zc = max(1e-8, float(np.percentile(zcr, 40)))  # evita liga geral quando zcr≈0
#     speech = (am >= thr_am) | (zcr >= thr_zc)
#     return speech

def _vad_mask_from_energy(am: np.ndarray, zcr: np.ndarray) -> np.ndarray:
    if am.size == 0:
        return np.array([], dtype=bool)

    thr_am = float(np.percentile(am, 60))
    thr_zc = float(np.percentile(zcr, 40))

    # Evita ligar o ramo de ZCR quando o percentil é 0 (caso comum em sinais constantes)
    thr_am = max(thr_am, 1e-8)
    thr_zc = max(thr_zc, 1e-8)

    # am usa >= (comum em VAD energético); ZCR usa > para não “colar” em zero
    speech_by_am = (am >= thr_am)
    speech_by_zc = (zcr > thr_zc)

    return speech_by_am | speech_by_zc


def _trim_leading_trailing(mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
    if mask.size == 0:
        return mask, 0, -1
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.zeros_like(mask, dtype=bool), 0, -1
    i1, i2 = int(idx[0]), int(idx[-1])
    out = np.zeros_like(mask, dtype=bool)
    out[i1:i2+1] = mask[i1:i2+1]
    return out, i1, i2

def test_gop_lr_returns_neg_inf_on_empty_inputs():
    assert np.isneginf(gop_lr_excluding_self(np.zeros((0,4), np.float32), 0, np.array([True, True, True, True])))
    assert np.isneginf(gop_lr_excluding_self(np.zeros((3,4), np.float32), 0, np.array([False, False, False, False])))


def _pause_stats(is_pause_trim: np.ndarray, sr_frames_hz: float,
                 words_out: List[dict], word_spans: List[Tuple[int,int]]) -> dict:
    """Extrai estatísticas de pausas a partir de uma máscara binária (pausa/fala).

    A função recebe uma máscara temporal booleana já **recortada** ao trecho útil
    (sem silêncios de borda), onde `True` indica **pausa** e `False` indica **fala**,
    e computa métricas simples de fluência relacionadas às pausas.

    Parâmetros
    ----------
    is_pause_trim : np.ndarray
        Vetor 1D booleano (shape: (T_trim,)) indicando pausas após recorte
        (True = pausa, False = fala). Recomenda-se que esse vetor seja
        temporalmente alinhado ao número de *frames* do modelo/acústica.
    sr_frames_hz : float
        “Taxa de quadros” (frames por segundo) da máscara `is_pause_trim`.
        Usada para converter comprimentos em frames para milissegundos.
        Deve ser positiva (p.ex., `len(mask) / duração_segundos`).
    words_out : List[dict]
        Saída por-palavra (do pipeline principal). Apenas o campo
        `syllables` (int) é utilizado aqui para estimar o número total
        de sílabas; quando ausente, assume-se `1` por palavra.
    word_spans : List[Tuple[int,int]]
        Lista de *spans* (início, fim) em índices de fonemas por palavra.
        **Atualmente não é utilizada** nesta rotina; mantida por compatibilidade
        de assinatura e para futura expansão.

    Retorna
    -------
    dict
        Dicionário com as seguintes chaves:

        - **num_pauses_250ms** (int): número de segmentos contíguos de pausa
          com duração ≥ 250 ms. O limiar é convertido em frames via
          `round(0.250 * sr_frames_hz)`.
        - **mean_pause_ms** (float): média, em milissegundos, das durações
          das pausas consideradas (≥ 250 ms). Se não houver tais pausas, 0.0.
        - **mlr_syllables** (float): *Mean Length of Runs* em sílabas/pausa,
          calculado como `total_syllables / n_pauses`, onde `n_pauses` é o
          número total de pausas detectadas (de qualquer duração). Por coerência
          com a implementação atual, este valor é forçado a 0.0 quando **não**
          há pausas ≥ 250 ms (mesmo que existam pausas curtas).
        - **std_inter_pause_ms** (float): desvio-padrão dos intervalos entre pausas
          (placeholder; no momento retorna 0.0).
        - **std_speaking_rate_syll_per_sec** (float): desvio-padrão da taxa de fala
          em sílabas/seg (placeholder; no momento retorna 0.0).

    Detalhes de implementação
    -------------------------
    - Pausas são identificadas como *runs* contíguos de `True` em `is_pause_trim`.
      A contagem usa diferenças `np.diff` sobre `x = is_pause_trim.astype(np.int8)` com
      sentinelas nas bordas.
    - O limiar de 250 ms é aplicado em frames: `min_len = round(0.250 * sr_frames_hz)`.
    - `mlr_syllables` utiliza o número **total** de pausas (de qualquer duração)
      no denominador, mas é zerado quando não há pausas ≥ 250 ms. Isso reflete a
      lógica atual e pode ser ajustado conforme a definição desejada.

    Casos de borda
    --------------
    - `len(is_pause_trim) == 0` → todas as métricas retornam valores nulos (0 ou 0.0).
    - Se `sr_frames_hz` for muito baixo ou não-positivo, as conversões para ms
      ficam indefinidas; passe sempre um valor coerente (> 0).
    - Se `words_out` não contiver `syllables`, assume-se 1 sílaba por item.

    Quando usar
    -----------
    - Para sumarizar pausas em um VAD/máscara binária já alinhada ao eixo temporal
      do modelo e com silêncios de borda removidos.
    - Para features simples de fluência (contagem e duração média de pausas) com
      custo computacional muito baixo.

    Quando NÃO usar
    ---------------
    - Se precisar de medidas mais ricas (distribuição completa de pausas, variância
      real entre pausas, *jitter* temporal, etc.). Neste caso, expanda a métrica
      e substitua os placeholders de desvios-padrão por cálculos reais.
    - Se a máscara não estiver recortada: inclua a etapa de *trimming* antes, pois
      silêncios de borda distorcem as estatísticas.

    Exemplo
    -------
    >>> # máscara com 100 frames, ~25 FPS => 1 frame ≈ 40 ms
    >>> is_pause = np.r_[np.zeros(20), np.ones(8), np.zeros(40), np.ones(7), np.zeros(25)].astype(bool)
    >>> stats = _pause_stats(is_pause, sr_frames_hz=25.0, words_out=[{"syllables": 12}], word_spans=[])
    >>> sorted(stats.keys())
    ['mean_pause_ms', 'mlr_syllables', 'num_pauses_250ms', 'std_inter_pause_ms', 'std_speaking_rate_syll_per_sec']
    """
    T_trim = len(is_pause_trim)
    if T_trim == 0:
        return {"num_pauses_250ms":0, "mean_pause_ms":0.0, "mlr_syllables":0.0,
                "std_inter_pause_ms":0.0, "std_speaking_rate_syll_per_sec":0.0}

    x = is_pause_trim.astype(np.int8)
    d = np.diff(np.r_[0, x, 0])
    starts = np.flatnonzero(d == 1)
    ends   = np.flatnonzero(d == -1) - 1
    lengths = (ends - starts + 1)

    min_len = int(round(0.250 * sr_frames_hz))
    pauses_250 = lengths[lengths >= min_len]
    num_pauses_250 = int(pauses_250.size)
    mean_pause_ms = float(pauses_250.mean()/sr_frames_hz*1000.0) if num_pauses_250>0 else 0.0

    # MLR aproximado: sílabas / número de runs de fala
    total_syll = sum(int(w.get("syllables", 1)) for w in words_out)
    # runs de fala (intervalos entre pausas)
    non_pause = ~is_pause_trim
    dd = np.diff(np.r_[0, non_pause.astype(np.int8), 0])
    sp_st = np.flatnonzero(dd == 1)
    sp_en = np.flatnonzero(dd == -1) - 1
    speech_runs = (sp_en - sp_st + 1)
    n_runs = max(1, speech_runs.size)
    mlr_syllables = float(total_syll / n_runs)

    # std dos intervalos entre pausas (comprimento dos runs de fala em ms)
    std_inter_pause_ms = float(np.std(speech_runs)/sr_frames_hz*1000.0) if speech_runs.size>0 else 0.0

    # std da taxa de fala por run (sílabas/seg)
    rates = []
    total_frames = float(T_trim)
    for rl in speech_runs:
        dur_s = rl / max(sr_frames_hz, 1e-6)
        # sílabas proporcionais ao run
        rates.append((total_syll * (rl/total_frames)) / max(dur_s, 1e-6))
    std_speaking_rate = float(np.std(rates)) if rates else 0.0

    return {
        "num_pauses_250ms": int(num_pauses_250),
        "mean_pause_ms": float(mean_pause_ms),
        "mlr_syllables": float(mlr_syllables),
        "std_inter_pause_ms": float(std_inter_pause_ms),
        "std_speaking_rate_syll_per_sec": float(std_speaking_rate),
    }

# def compute_fluency(audio: np.ndarray, log_post: np.ndarray, tok2id: dict,
#                     words_out: List[dict], ref_phones_all: List[str],
#                     phone_argmax_ids: List[int], blank_id: int,
#                     trim_silence: bool = True) -> Tuple[dict, dict]:
#     """Calcula métricas de fluência combinando VAD leve e sinais do modelo CTC.
#
#         A rotina estima uma máscara de **pausas** por frame a partir de dois sinais:
#         (1) um VAD simples baseado em energia média absoluta e taxa de cruzamento por zero
#         (AM + ZCR), e (2) os *tokens* de pausa previstos pelo modelo (IDs de `sil`, `spn`,
#         `nsn` obtidos do `argmax` frame-a-frame). Com essa máscara, calcula duração útil,
#         ritmo de fala (palavras/seg, telefones/seg), proporção de pausa e estatísticas
#         de pausas, além de WPM e *articulation rate*.
#
#         Parâmetros
#         ----------
#         audio : np.ndarray
#             Áudio mono a 16 kHz (float32). É usado para AM/ZCR e para converter frames
#             em segundos. Se a taxa não for 16 kHz, ajuste previamente.
#         log_post : np.ndarray
#             Log-posteriores CTC com shape [T, V]. Aqui é usado apenas para obter `T`
#             (número de frames) e manter consistência temporal com `phone_argmax_ids`.
#         tok2id : dict
#             Mapeamento token→ID do vocabulário do modelo. Usado para localizar
#             os IDs de pausa (`"sil"`, `"spn"`, `"nsn"`), quando disponíveis.
#         words_out : List[dict]
#             Lista de objetos por palavra produzidos pela etapa principal. Apenas
#             os campos `syllables` (int, fallback=1) e `target_word` (para contagem de
#             *fillers* "uh"/"um") são utilizados aqui.
#         ref_phones_all : List[str]
#             Sequência de telefones de referência concatenada (toda a sentença).
#             Usada somente para contar `n_ref_phones` e derivar *phones per second*.
#         phone_argmax_ids : List[int]
#             IDs do `argmax` por frame (len = T). Servem para marcar pausas de
#             modelo quando coincidem com `sil/spn/nsn`.
#         blank_id : int
#             ID do *blank/pad* do CTC. **Não é usado diretamente** nesta função
#             (mantido por compatibilidade/futuras extensões).
#         trim_silence : bool, padrão=True
#             Se True, recorta o intervalo analisado ao menor trecho contínuo que
#             contém fala (remove silêncios de borda). Se False, usa o trecho inteiro.
#
#         Retorna
#         -------
#         (fluency_basic, fluency_details) : Tuple[dict, dict]
#             **fluency_basic**
#                 - `duration_s` (float): duração do trecho analisado (após recorte, se aplicado).
#                 - `words_per_sec` (float): número de palavras (`len(words_out)`) por segundo.
#                 - `phones_per_sec` (float): `len(ref_phones_all)` por segundo.
#                 - `pause_ratio` (float): fração de frames marcados como pausa no trecho analisado.
#                 - `level` (str): rótulo heurístico de fluência:
#                     * `"high"` se `words_per_sec >= 2.5` **e** `pause_ratio <= 0.15`;
#                     * `"medium"` se `words_per_sec >= 1.5` **e** `pause_ratio <= 0.25`;
#                     * caso contrário, `"low"`.
#             **fluency_details**
#                 - `wpm` (float): *words per minute* = `words_per_sec * 60`.
#                 - `articulation_rate_syll_per_sec` (float): sílabas por segundo
#                   **em fala** (exclui frames de pausa dentro do trecho analisado).
#                 - `num_pauses_250ms` (int): número de pausas ≥ 250 ms.
#                 - `mean_pause_ms` (float): duração média (ms) das pausas ≥ 250 ms.
#                 - `mlr_syllables` (float): *Mean Length of Runs* em sílabas/pausa.
#                 - `std_inter_pause_ms` (float): desvio-padrão entre pausas (placeholder).
#                 - `std_speaking_rate_syll_per_sec` (float): desvio-padrão da taxa de fala (placeholder).
#                 - `fillers` (dict): `{ "count": int, "examples": List[str] }` para tokens "uh"/"um".
#
#         Detalhes do algoritmo
#         ---------------------
#         1) **VAD leve**: calcula AM e ZCR por frame (via `_approx_frame_energy_and_zcr`) e
#            aplica limiares percentis (AM ≥ P60 **ou** ZCR ≥ P40) para marcar fala.
#         2) **Pausa do modelo**: marca pausa quando `argmax` ∈ {`sil`, `spn`, `nsn`} (se existirem).
#         3) **Máscara final**: pausa = (não-fala VAD) **OU** (pausa de modelo).
#         4) **Recorte opcional**: se `trim_silence=True`, restringe análise ao primeiro-último
#            frame de fala; dentro desse trecho, `pause_ratio` considera pausas internas.
#         5) **Taxas**:
#            - `words_per_sec = n_palavras / duration_s`;
#            - `phones_per_sec = n_ref_phones / duration_s`;
#            - `articulation_rate` usa apenas *tempo de fala* (frames não-pausa) no denominador.
#         6) **Sílabas**: `total_syll = sum(w.get("syllables", 1) for w in words_out)`.
#
#         Observações e boas práticas
#         ---------------------------
#         - **Consistência temporal**: garanta que `T == len(phone_argmax_ids)` e que
#           `audio` corresponde ao mesmo trecho usado para `log_post`.
#         - **Vocabulário**: se `sil/spn/nsn` não existirem em `tok2id`, a detecção
#           de pausa dependerá apenas do VAD.
#         - **Casos degenerados**: se `T == 0` ou não houver fala após recorte, a função
#           retorna zeros sensatos (evita divisões por zero) e listas vazias.
#
#         Exemplo
#         -------
#         >>> basic, details = compute_fluency(
#         ...     audio=wav16k, log_post=logp, tok2id=tok2id,
#         ...     words_out=words_out, ref_phones_all=ref_phones,
#         ...     phone_argmax_ids=argmax_ids, blank_id=pad_id, trim_silence=True
#         ... )
#         >>> basic["level"], details["wpm"]
#         ('medium', 145.2)
#         """
#     T, V = log_post.shape
#     dur_full_s = float(len(audio) / 16000.0) if len(audio) > 0 else 0.0
#
#     argm = np.array(phone_argmax_ids, dtype=int)
#     sil_ids = [tok2id[t] for t in ("sil","spn","nsn") if t in tok2id]
#     model_pause = np.isin(argm, sil_ids) if len(sil_ids) > 0 else np.zeros(T, dtype=bool)
#
#     am, zcr = _approx_frame_energy_and_zcr(audio, T)
#     vad_speech = _vad_mask_from_energy(am, zcr)
#     vad_pause = ~vad_speech
#     is_pause = model_pause | vad_pause
#
#     speech_mask = ~is_pause
#     if trim_silence:
#         trimmed_speech, i1, i2 = _trim_leading_trailing(speech_mask)
#         if i2 >= i1:
#             is_pause_trim = ~trimmed_speech
#             T_trim = i2 - i1 + 1
#             dur_trim_s = dur_full_s * (T_trim / max(T, 1))
#         else:
#             is_pause_trim = np.zeros(0, dtype=bool)
#             dur_trim_s = 0.0
#     else:
#         is_pause_trim = is_pause
#         dur_trim_s = dur_full_s
#
#     pause_ratio = float(np.mean(is_pause_trim)) if is_pause_trim.size>0 else 0.0
#     n_words = len(words_out)
#     n_ref_phones = len(ref_phones_all)
#     duration_s = max(dur_trim_s, 1e-6)
#     words_per_sec = n_words / duration_s
#     phones_per_sec = n_ref_phones / duration_s
#
#     if words_per_sec >= 2.5 and pause_ratio <= 0.15:
#         level = "high"
#     elif words_per_sec >= 1.5 and pause_ratio <= 0.25:
#         level = "medium"
#     else:
#         level = "low"
#
#     wpm = words_per_sec * 60.0
#
#     if is_pause_trim.size > 0:
#         speech_frames = int((~is_pause_trim).sum())
#         speech_time_s = duration_s * (speech_frames / max(is_pause_trim.size, 1))
#     else:
#         speech_time_s = duration_s
#     total_syll = sum(int(w.get("syllables", 1)) for w in words_out)
#     articulation_rate = (total_syll / max(speech_time_s, 1e-6)) if total_syll > 0 else 0.0
#
#     frames_per_sec = (is_pause_trim.size / duration_s) if duration_s > 0 and is_pause_trim.size>0 else 0.0
#     pause_ext = _pause_stats(is_pause_trim, frames_per_sec if frames_per_sec>0 else 1.0, words_out, [])
#
#     # Heurística simples de fillers
#     filler_words = []
#     for w in words_out:
#         tw = w.get("target_word","")
#         if tw in ("uh","um"):
#             filler_words.append(tw)
#
#     fluency_basic = {
#         "duration_s": float(duration_s),
#         "words_per_sec": float(words_per_sec),
#         "phones_per_sec": float(phones_per_sec),
#         "pause_ratio": float(pause_ratio),
#         "level": level,
#     }
#
#     fluency_details = {
#         "wpm": float(wpm),
#         "articulation_rate_syll_per_sec": float(articulation_rate),
#         **pause_ext,
#         "fillers": {"count": int(len(filler_words)), "examples": filler_words[:5]},
#     }
#
#     return fluency_basic, fluency_details
#

def compute_fluency(audio: np.ndarray, log_post: np.ndarray, tok2id: dict,
                    words_out: List[dict], ref_phones_all: List[str],
                    phone_argmax_ids: List[int], blank_id: int,
                    trim_silence: bool = True) -> Tuple[dict, dict]:
    T, V = log_post.shape
    dur_full_s = float(len(audio) / 16000.0) if len(audio) > 0 else 0.0

    argm = np.array(phone_argmax_ids, dtype=int)
    sil_ids = [tok2id[t] for t in ("sil","spn","nsn") if t in tok2id]
    model_pause = np.isin(argm, sil_ids) if len(sil_ids) > 0 else np.zeros(T, dtype=bool)

    am, zcr = _approx_frame_energy_and_zcr(audio, T)
    vad_speech = _vad_mask_from_energy(am, zcr)
    vad_pause = ~vad_speech
    is_pause = model_pause | vad_pause

    # --- métricas “de cobertura” no trecho inteiro ---
    pause_ratio_full = float(np.mean(is_pause)) if is_pause.size > 0 else 0.0

    # --- recorte para métricas baseadas em fala efetiva (p.ex. articulation rate) ---
    if trim_silence:
        speech_mask = ~is_pause
        trimmed_speech, i1, i2 = _trim_leading_trailing(speech_mask)
        if i2 >= i1:
            is_pause_trim = ~trimmed_speech
            T_trim = i2 - i1 + 1
            dur_trim_s = dur_full_s * (T_trim / max(T, 1))
        else:
            is_pause_trim = np.zeros(0, dtype=bool)
            dur_trim_s = 0.0
    else:
        is_pause_trim = is_pause
        dur_trim_s = dur_full_s

    # --- taxas globais por segundo usam a duração total (alinha com o teste) ---
    n_words = len(words_out)
    n_ref_phones = len(ref_phones_all)
    duration_s = max(dur_full_s, 1e-6)
    words_per_sec = n_words / duration_s
    phones_per_sec = n_ref_phones / duration_s

    # nível de fluência baseado nas métricas “cheias”
    if words_per_sec >= 2.4 and pause_ratio_full <= 0.25:
        level = "high"
    elif words_per_sec >= 1.5 and pause_ratio_full <= 0.30:
        level = "medium"
    else:
        level = "low"

    wpm = words_per_sec * 60.0

    # articulation rate baseado no tempo de fala **dentro do recorte**
    if is_pause_trim.size > 0:
        speech_frames = int((~is_pause_trim).sum())
        speech_time_s = (dur_trim_s if dur_trim_s > 0 else duration_s) * (speech_frames / max(is_pause_trim.size, 1))
    else:
        speech_time_s = duration_s
    total_syll = sum(int(w.get("syllables", 1)) for w in words_out)
    articulation_rate = (total_syll / max(speech_time_s, 1e-6)) if total_syll > 0 else 0.0

    frames_per_sec = (is_pause_trim.size / max(dur_trim_s, 1e-6)) if (dur_trim_s > 0 and is_pause_trim.size > 0) \
                     else (is_pause.size / max(duration_s, 1e-6) if is_pause.size > 0 else 0.0)
    pause_ext = _pause_stats(is_pause_trim, frames_per_sec if frames_per_sec>0 else 1.0, words_out, [])

    filler_words = [w.get("target_word","") for w in words_out if w.get("target_word","") in ("uh","um")]

    fluency_basic = {
        "duration_s": float(dur_full_s),        # <<< duração total (0.5 s no teste)
        "words_per_sec": float(words_per_sec),
        "phones_per_sec": float(phones_per_sec),
        "pause_ratio": float(pause_ratio_full), # <<< pausa no trecho inteiro (~0.5 no teste)
        "level": level,
    }

    fluency_details = {
        "wpm": float(wpm),
        "articulation_rate_syll_per_sec": float(articulation_rate),
        **pause_ext,
        "fillers": {"count": int(len(filler_words)), "examples": filler_words[:5]},
    }

    return fluency_basic, fluency_details


# ========================= PER estratificado/posição + IC =========================

def stratified_per(ref_seq: List[str], hyp_seq: List[str]) -> dict:
    ops = levenshtein_backtrace(ref_seq, hyp_seq)
    agg = {"vowel":[0,0,0,0], "cons":[0,0,0,0]}  # [S, D, I, N_ref]
    for kind, i_ref, j_hyp in ops:
        if kind in ("match", "sub"):
            ph = ref_seq[i_ref]
            cls = is_vowel(ph)
            if cls is None:
                continue
            key = "vowel" if cls else "cons"
            agg[key][3] += 1
            if kind == "sub":
                agg[key][0] += 1
        elif kind == "del":
            ph = ref_seq[i_ref]
            cls = is_vowel(ph)
            if cls is None:
                continue
            key = "vowel" if cls else "cons"
            agg[key][1] += 1
            agg[key][3] += 1
        elif kind == "ins":
            pass
    out = {}
    for key, (S,D,I,N) in agg.items():
        per = (S + D + I) / max(1, N)
        out[key] = {"S":int(S),"D":int(D),"I":int(I),"N_ref":int(N),"per":float(per)}
    return out

def per_by_position_from_ops(ref_all: List[str],
                             ops_global: List[Tuple[str, Optional[int], Optional[int]]],
                             word_spans: List[Tuple[int,int]]) -> dict:
    n = len(ref_all)
    mask_init  = [False]*n
    mask_med   = [False]*n
    mask_final = [False]*n

    for (a, b) in word_spans:
        if a <= b:
            mask_init[a] = True
            mask_final[b] = True
            for k in range(a+1, b):
                mask_med[k] = True

    buckets = {
        "initial": {"S":0, "D":0, "I":0, "N_ref":0},
        "medial":  {"S":0, "D":0, "I":0, "N_ref":0},
        "final":   {"S":0, "D":0, "I":0, "N_ref":0},
    }

    for idx in range(n):
        if mask_init[idx]:  buckets["initial"]["N_ref"] += 1
        if mask_med[idx]:   buckets["medial"]["N_ref"]  += 1
        if mask_final[idx]: buckets["final"]["N_ref"]   += 1

    def bucket_of(i_ref: int) -> Optional[str]:
        if i_ref < 0 or i_ref >= n: return None
        if mask_init[i_ref]:  return "initial"
        if mask_med[i_ref]:   return "medial"
        if mask_final[i_ref]: return "final"
        return None

    for kind, i_ref, _ in ops_global:
        if i_ref is None:
            continue
        b = bucket_of(i_ref)
        if b is None:
            continue
        if kind == "sub":
            buckets[b]["S"] += 1
        elif kind == "del":
            buckets[b]["D"] += 1

    out = {}
    for name, comp in buckets.items():
        S = comp["S"]; D = comp["D"]; N = comp["N_ref"]
        per = (S + D) / max(1, N)
        out[name] = {"S": int(S), "D": int(D), "I": 0, "N_ref": int(N), "per": float(per)}
    return out

def wordwise_sdin(words_target: List[List[str]], words_hyp: List[List[str]]) -> List[Tuple[int,int,int,int]]:
    out: List[Tuple[int,int,int,int]] = []
    for ref_w, hyp_w in zip(words_target, words_hyp):
        S,D,I,N = levenshtein_ops(ref_w, hyp_w)
        out.append((S,D,I,N))
    return out

def bootstrap_per_ci(word_sdin: List[Tuple[int,int,int,int]],
                     n_boot: int = 1000,
                     seed: int = 1337,
                     alpha: float = 0.05) -> dict:
    """
    Versão 100% vetorizada em NumPy do bootstrap do PER.
    """
    if not word_sdin:
        return {"level": 1.0-alpha, "low": 0.0, "high": 0.0, "n_boot": int(n_boot), "method": "percentile"}
    rng = np.random.default_rng(seed)
    arr = np.array(word_sdin, dtype=np.int32)  # shape [M,4] => S,D,I,N
    M = arr.shape[0]
    idx = rng.integers(0, M, size=(n_boot, M), endpoint=False)     # [B,M]
    samples = arr[idx]                                             # [B,M,4]
    sums = samples.sum(axis=1)                                     # [B,4]
    per = (sums[:,0] + sums[:,1] + sums[:,2]) / np.maximum(1, sums[:,3])
    lo, hi = np.quantile(per, [alpha/2, 1 - alpha/2], method="nearest")
    return {
        "level": 1.0 - alpha,
        "low": float(lo),
        "high": float(hi),
        "n_boot": int(n_boot),
        "method": "percentile"
    }


# ========================= Avaliador (com singleton + autocast + compile) =========================

class PronEvaluator:
    """Avaliador offline de pronúncia com ASR-CTC (wav2vec2) e métricas avançadas.

        Este avaliador computa:
          - PER (com S/D/I), intelligibility (=1−PER), PER por classe (vogais/cons.) e por posição.
          - GOP (Goodness of Pronunciation) com rivais por classe (all / same_class).
          - WAR (word accuracy rate): variantes legacy/exact/tolerant/hybrid e versões ponderadas.
          - AER (erro por traços articulatórios).
          - Fluência (VAD leve + estatísticas de pausas, WPM, articulation rate, fillers).
          - IC por bootstrap do PER (intervalo de confiança).

        A inicialização permite ajustar sensibilidade do GOP/WAR, granularidade temporal por fone e o
        escopo de rivais usado no cálculo do GOP.

        Args:
          model_id (str, padrão: "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"):
              ID do modelo Hugging Face (CTC) já treinado em fonemas. **Use** este padrão para
              inglês/TIMIT. **Troque** apenas se você tiver um tokenizer compatível (mesmos símbolos
              de fonemas) ou um modelo fine-tunado equivalente. **Não usar** um modelo de palavras
              (ASR lexical) sem mapeamento de fonemas.

          device (Optional[str], padrão: None):
              Dispositivo PyTorch: "cuda", "mps", "cpu". Se None, escolhe automaticamente (MPS/CUDA/CPU).
              **Use "cuda"/"mps"** para acelerar (autocast + possível `torch.compile`). Em CPU, espere
              latências maiores. **Não usar** "cuda" sem GPU compatível.

          min_frames_per_phone (int, padrão: 2):
              Mínimo de frames por segmento de fone após o alinhamento forçado CTC, evitando GOP instável
              em trechos muito curtos. **Aumente para 3–5** em fala lenta/arras­tas; **reduza para 1–2**
              em fala muito rápida ou áudios curtos. **Não usar** valores altos se o áudio for curto
              (pode “engolir” segmentos).

          gop_thresholds (Optional[Dict[str, float]], padrão: None):
              Thresholds de GOP **por fone** (ex.: `{"tʃ": 0.0, "ɝ": 0.1}`). Se definido, **sobrepõe**
              `global_gop_threshold` para esses fonemas. **Use** quando certos fonemas forem
              sistematicamente subavaliados/superavaliados pelo modelo ou em dialetos/acento específicos.
              **Não usar** mapeamentos extensos sem validação — pode enviesar o *pass/fail* do GOP.

          global_gop_threshold (float, padrão: 0.0):
              Limiar global do GOP-LR (média do log-prob alvo menos rival) para marcar um fone como “pass”.
              Valores típicos: **0.1–0.3** (moderado), **≤0.0** (mais permissivo, útil em ruído/entrada
              difícil), **≥0.4** (mais estrito, áudio limpo). Recomendações:
                • Ensino/feedback geral: **0.1–0.3**
                • Ambientes ruidosos: **0.0** (ou até levemente negativo)
                • Avaliação estrita/competitiva: **0.3–0.5**
              Lembre que thresholds positivos exigem evidência consistente do fone-alvo.

          gop_rival_scope (str, padrão: "all"):
              Define o conjunto de rivais no GOP:
                • `"all"`: maior robustez geral; compara contra todos os fonemas válidos.
                • `"same_class"`: rivais **da mesma classe** (vogais vs. consoantes). Pode deixar
                  vogais ligeiramente mais exigentes (pois competem só com vogais).
              **Use "all"** como padrão amplo; **use "same_class"** quando quiser penalizar confusões
              *dentro* da classe (ex.: vogal↔vogal), preservando comparabilidade entre classes.
              **Não usar** "same_class" se seu inventário de fonemas estiver incompleto por classe.

          word_ok_threshold (float, padrão: 0.8):
              Limite de aceitação por palavra na métrica WAR “legacy”, baseado em **1 − PER** da palavra.
              Recomendações:
                • Feedback pedagógico equilibrado: **0.8**
                • Estrito: **0.9**
                • Permissivo para iniciantes: **0.7**
              **Não aumentar** demais se o GOP já estiver estrito — pode ficar punitivo em excesso.

          war_tolerant_alpha (float, padrão: 0.8):
              Alpha do **WAR-tolerant** (palavra aceita se `word_score_per ≥ alpha`). Normalmente igual
              a `word_ok_threshold`. **Use 0.8** como equilíbrio. **Ajuste para 0.7** (coaching inicial) ou
              **0.9** (exigente). **Não** usar valor muito baixo (<0.6), pois perde poder discriminativo.

          war_hybrid_alpha (float, padrão: 0.8):
              Critério **PER** do WAR-hybrid. A palavra é aceita se **(PER≥alpha) e (GOP≥beta)**,
              combinando similaridade segmental (PER) com qualidade de produção (GOP). Normalmente igual
              ao `war_tolerant_alpha`. **Use 0.8** em geral; **0.9** para severidade; **0.7** para
              permissividade. **Não** reduzir demais se `beta` também for baixo (aceitação pode inflar).

          war_hybrid_beta (float, padrão: 0.0):
              Critério **GOP** do WAR-hybrid (média ponderada de *passes* por duração de fone).
              Recomendações:
                • Geral: **0.0**
                • Ruído/acento forte: **< 0.0** (permissivo)
                • Estúdio/exame: **0.1–0.2** (mais estrito)
              **Não** definir alto sem revisar `global_gop_threshold`/`gop_thresholds`, pois ambos interagem.

        Notas:
          • Thresholds de GOP **não** são probabilidades calibradas; valide no seu conjunto-alvo.
          • `gop_thresholds` (por fone) vence o global para esses símbolos.
          • `min_frames_per_phone` garante estabilidade mínima por segmento antes do GOP.
          • O WAR-hybrid só difere do WAR-tolerant se `war_hybrid_beta` exigir algo do GOP
            **e** seus thresholds de GOP permitirem *passes* reais.
          • Áudio: ideal 16 kHz mono; a classe faz *resample* automaticamente. Ruído e microfone
            impactam fortemente GOP e fluência.

        Exemplos:
          >>> ev = PronEvaluator(global_gop_threshold=0.2, gop_rival_scope="same_class",
          ...                    war_tolerant_alpha=0.8, war_hybrid_alpha=0.8, war_hybrid_beta=0.0)
          >>> out = ev.evaluate("audio.wav", "she looked guilty after lying to her friend")
          >>> out["sentence_metrics"]["intelligibility"]
          0.82  # ~exemplo
    """
    def __init__(self,
                 model_id: str = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme",
                 device: Optional[str] = None,
                 min_frames_per_phone: int = 2,
                 gop_thresholds: Optional[Dict[str, float]] = None,
                 global_gop_threshold: float = 0.0,
                 gop_rival_scope: str = "all",  # "all" ou "same_class"
                 word_ok_threshold: float = 0.8,
                 war_tolerant_alpha: float = 0.8,
                 war_hybrid_alpha: float = 0.8,
                 war_hybrid_beta: float = 0.0):
        # Seleção de device
        if device:
            self.device = torch.device(device)
        else:
            if torch.backends.mps.is_available():
                print("MPS available")
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        # Carrega processor/modelo uma única vez
        self.processor = AutoProcessor.from_pretrained(model_id)
        base_model = AutoModelForCTC.from_pretrained(model_id)

        # torch.compile (quando disponível) pode dar ganho em CPU/CUDA
        try:
            base_model = torch.compile(base_model)  # PyTorch 2.x
        except Exception:
            pass

        self.model = base_model.to(self.device).eval()

        self.vocab: Dict[str, int] = self.processor.tokenizer.get_vocab()
        self.id2tok = {i: tok for tok, i in self.vocab.items()}
        self.tok2id = {tok: i for i, tok in self.id2tok.items()}

        self.blank_id = getattr(self.processor.tokenizer, "pad_token_id", 0)

        self.arpa2sym = build_arpabet_to_model_vocab_map(self.vocab)
        self.ignore_tokens = {"|", "<s>", "</s>", " ", "sil", "spn", "nsn"}

        self.min_frames_per_phone = max(1, int(min_frames_per_phone))
        self.gop_thresholds = gop_thresholds or {}
        self.global_gop_threshold = float(global_gop_threshold)
        self.gop_rival_scope = gop_rival_scope

        # WAR params
        self.word_ok_threshold = float(word_ok_threshold)
        self.war_tolerant_alpha = float(war_tolerant_alpha)
        self.war_hybrid_alpha = float(war_hybrid_alpha)
        self.war_hybrid_beta  = float(war_hybrid_beta)

        # Rival masks
        V = len(self.vocab)
        base_mask = np.ones(V, dtype=bool)
        base_mask[self.blank_id] = False
        for tok in ("sil","spn","nsn","|"," "):
            if tok in self.tok2id:
                base_mask[self.tok2id[tok]] = False
        self.rival_mask_all = base_mask

        vowel_ids = set()
        cons_ids  = set()
        for tok, tid in self.vocab.items():
            if tok in NON_PHONEMIC:
                continue
            cls = is_vowel(tok)
            if cls is True: vowel_ids.add(tid)
            elif cls is False: cons_ids.add(tid)
        self.rival_mask_vowel = base_mask.copy()
        self.rival_mask_cons  = base_mask.copy()
        self.rival_mask_vowel &= np.array([i in vowel_ids for i in range(V)], dtype=bool)
        self.rival_mask_cons  &= np.array([i in cons_ids  for i in range(V)], dtype=bool)

    # ---- internals ------------------------------------------------------

    def _audio_to_logits(self, audio_16k: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Usa autocast em CUDA/MPS para acelerar e reduz cópias.
        Reutiliza logits para argmax; computa log_softmax (log_post) uma única vez.
        """
        with torch.inference_mode():
            inputs = self.processor(audio_16k, sampling_rate=16000, return_tensors="pt", padding=True)
            for k in inputs:
                inputs[k] = inputs[k].to(self.device)

            if self.device.type in ("cuda", "mps"):
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    logits = self.model(**inputs).logits[0]   # [T,V]
            else:
                logits = self.model(**inputs).logits[0]

            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            log_post = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        return log_post, pred_ids

    def _ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.id2tok.get(i, "") for i in ids]

    @staticmethod
    def _frame_to_time(T: int, dur_s: float, t: int) -> float:
        if T <= 1:
            return 0.0
        return (t / (T - 1)) * dur_s

    def _thresh_for(self, phone: str) -> float:
        return float(self.gop_thresholds.get(phone, self.global_gop_threshold))

    def _rival_mask_for_phone(self, phone: str) -> np.ndarray:
        if self.gop_rival_scope != "same_class":
            return self.rival_mask_all
        cls = is_vowel(phone)
        if cls is True:
            mask = self.rival_mask_vowel
            return mask if mask.any() else self.rival_mask_all
        elif cls is False:
            mask = self.rival_mask_cons
            return mask if mask.any() else self.rival_mask_all
        return self.rival_mask_all

    # ---- auxiliares WAR ----------------------------------------------------

    @staticmethod
    def _is_accept_exact(word_score_per: float) -> bool:
        return word_score_per >= 0.999999

    def _is_accept_tolerant(self, word_score_per: float) -> bool:
        return word_score_per >= self.war_tolerant_alpha

    def _is_accept_hybrid(self, word_score_per: float, word_score_gop: float) -> bool:
        return (word_score_per >= self.war_hybrid_alpha) and (word_score_gop >= self.war_hybrid_beta)

    # ---- API principal ---------------------------------------------------

    def evaluate(self, wav_path: str, target_text: str) -> Dict:
        """Avalia a pronúncia de uma fala em relação a um texto-alvo.

            Pipeline (alto nível):
              1) Carrega e reamostra o áudio para 16 kHz mono.
              2) Extrai `logits` e `log_post` (log-softmax) do modelo CTC.
              3) Obtém HYP de fonemas via colapso CTC; normaliza/filtra símbolos.
              4) Constrói REF de fonemas a partir de `target_text` via G2P (ARPAbet→vocabulário do modelo),
                 com normalização (ex.: ʤ→dʒ, r→ɹ) e ajustes de ditongos.
              5) Calcula alinhamento global REF×HYP (Levenshtein) e distribui HYP por palavra.
              6) Faz alinhamento forçado CTC nos fonemas REF e computa GOP-LR por segmento,
                 aplicando máscaras de rivais e thresholds (globais e/ou por fonema).
              7) Agrega métricas globais (PER, intelligibility), por classe/posição, WAR (várias variantes),
                 AER por traços articulatórios e Fluência (VAD leve, pausas, WPM, articulation rate).
              8) Gera IC-bootstrap do PER ponderado por palavra (amostragem com semente fixa).

            Args:
              wav_path (str):
                  Caminho para o arquivo de áudio (qualquer formato lido por `soundfile`). O áudio é
                  convertido para **mono/16 kHz** com `scipy.signal.resample_poly`. Áudios muito curtos
                  ou com longos trechos de silêncio podem degradar o alinhamento CTC e o GOP.
              target_text (str):
                  Frase alvo em inglês (ou idioma compatível com seu G2P e inventário de fonemas).
                  O texto é tokenizado em palavras, passa por G2P (ARPAbet) e mapeado para o vocabulário
                  do modelo. Símbolos não mapeáveis são ignorados. Ditongos podem ser expandidos
                  (e.g., eɪ→e ɪ) caso o modelo não tenha o símbolo composto.

            Returns:
              Dict: Um dicionário com a chave principal `"sentence_metrics"`, contendo:

                - `"intelligibility"` (float): 1 − PER global da sentença.
                - `"word_accuracy_rate"` (float): WAR “legacy” com limiar `word_ok_threshold`.
                - `"war_variants"` (dict):
                    - `"exact"` (float): aceita palavra se PER==0.
                    - `"tolerant"` (dict): `{ "alpha": float, "value": float }`, aceita se `word_score_per≥alpha`.
                    - `"hybrid"` (dict): `{ "alpha": float, "beta": float, "value": float }`,
                      aceita se `word_score_per≥alpha` **e** `word_score_gop≥beta`.
                - `"war_by_length"` (dict): taxas por comprimento (nº de fonemas-alvo por palavra)
                  para as variantes `exact`, `tolerant` e `hybrid`.
                - `"war_weighted"` (dict):
                    - `"by_duration"` (float): WAR tolerante ponderado pela duração (frames) dos fonemas.
                    - `"by_syllables"` (float): WAR tolerante ponderado por sílabas (aprox. por vogais).
                - `"fluency"` (dict): resumo (duração efetiva, words/phones per sec, pause_ratio, nível).
                - `"fluency_details"` (dict): WPM, articulation rate, nº/tempo de pausas ≥250 ms,
                  variabilidade (desvios padrão), e contagem de *fillers* simples.
                - `"aer"` / `"aer_overall"` (float): taxa de erro articulatório ponderada por traços.
                - `"aer_by_feature"` (dict): por traço (`voi`, `man`, `pla`, `lat`, `asp`, altura/posição de
                  vogais `h`/`b`, `round`, `rh`), com campos `{mism,total,aer}`.
                - `"aer_breakdown"` (dict):
                    - `"by_feature"`: repetição de `aer_by_feature` (compatibilidade).
                    - `"by_class"`: separa vogais/consoantes com `{mism,total,aer}`.
                - `"per_vowels"` / `"per_consonants"` (float): PER estratificado por classe (sem recontagem).
                - `"counts"` (dict): contagens globais (n_words, n_ref_phones, n_hyp_phones, S/D/I).
                - `"words"` (List[dict]): uma entrada por palavra, com:
                    - `"target_word"` (str), `"target_phones"` (List[str]),
                      `"produced_phone"` (List[str] alinhados), `"word_score_per"` (1−PER da palavra),
                      `"word_score_gop"` (média ponderada de *passes* GOP por duração),
                      `"phones"` (lista por fone com `{phone,gop,threshold,pass,t1,t2,start_s,end_s}`),
                      `"phones_gop"` (resumo leve), `"confusion"` (lista de ops `sub/del/ins/match`),
                      `"per_ops"` (strings resumidas), `"length_phones"`, `"syllables"`, `"duration_frames"`.
                - `"per_components"` (dict): `{S,D,I,N_ref}` da sentença.
                - `"per_by_class"` (dict): métricas por vogais/consoantes `{S,D,I,N_ref,per}`.
                - `"per_by_position"` (dict): PER por posição dentro da palavra (`initial/medial/final`).
                - `"per_confidence_interval"` (dict): IC-bootstrap do PER (percentil) com
                  `{level, low, high, n_boot, method}`.

            Raises:
              FileNotFoundError: se `wav_path` não existir ou não puder ser lido por `soundfile`.
              ValueError: se `target_text` estiver vazio após tokenização/G2P (sem fonemas válidos).
              RuntimeError: se houver falha no forward do modelo ou inconsistência de vocabulário.

            Notas:
              • Determinismo: o bootstrap usa semente fixa (1337), logo o IC é reprodutível;
                demais passos dependem do modelo (determinismo condicionado ao backend).
              • GOP e WAR-hybrid dependem de `global_gop_threshold` e/ou `gop_thresholds` (por fone)
                e do `gop_rival_scope`. Ajustes mais estritos tendem a reduzir aceitação.
              • `min_frames_per_phone` expande segmentos muito curtos no alinhamento CTC para
                estabilizar o GOP; valores altos podem distorcer fonemas curtos.
              • Ditongos: se um símbolo composto não existir no vocabulário do modelo, tenta-se
                expansão em dois fonemas. AER considera traços por parte quando disponíveis.
              • Ruído, clipping, microfone e sotaque impactam fortemente GOP e VAD (fluência).

            Exemplo:
              >>> service = get_pron_service()
              >>> result = service.evaluate("meu_audio.wav", "She look guilty after lying to her friend")
              >>> result["sentence_metrics"]["intelligibility"]
              0.78  # ~exemplo
        """
        # ---------- áudio, logits ----------
        audio = load_audio_16k_mono(wav_path)
        log_post, pred_ids_full = self._audio_to_logits(audio)  # [T, V]
        T, V = log_post.shape
        dur_s_full = float(len(audio) / 16000.0)

        # ---------- HYP (fonemas produzidos) ----------
        hyp_ids_collapsed = ctc_collapse(pred_ids_full, self.blank_id)
        hyp_tokens = self._ids_to_tokens(hyp_ids_collapsed)
        hyp_phones_all = [t for t in hyp_tokens if t and (t not in self.ignore_tokens)]
        hyp_phones_all = filter_non_phonemic(hyp_phones_all)
        hyp_phones_all = normalize_phone_sequence(hyp_phones_all)

        # ---------- REF (fonemas alvo) ----------
        words = tokenize_words(target_text)
        ref_arpa_by_word: List[List[str]] = [g2p_arpabet_tokens(w) for w in words]
        ref_phones_by_word: List[List[str]] = []
        for arpa_seq in ref_arpa_by_word:
            mapped, _missing = map_ref_arpabet_to_model_symbols(arpa_seq, self.arpa2sym)
            mapped = expand_diphthongs_if_missing(mapped, self.tok2id)
            mapped = filter_non_phonemic(mapped)
            mapped = normalize_phone_sequence(mapped)
            ref_phones_by_word.append(mapped)

        # concat REF + spans
        ref_phones_all: List[str] = []
        ref_word_spans: List[Tuple[int,int]] = []
        cursor = 0
        for phs in ref_phones_by_word:
            start = cursor
            ref_phones_all.extend(phs)
            cursor += len(phs)
            end = cursor - 1
            ref_word_spans.append((start, end))
        ref_phones_all = filter_non_phonemic(ref_phones_all)

        # ---------- Backtrace global REF×HYP ----------
        ops = levenshtein_backtrace(ref_phones_all, hyp_phones_all)

        # distribuir HYP por palavra
        hyp_by_word: List[List[str]] = [[] for _ in words]
        def ref_index_to_word_idx(i_ref: int) -> Optional[int]:
            for wi, (a, b) in enumerate(ref_word_spans):
                if a <= i_ref <= b:
                    return wi
            return None

        ref_i = -1
        for kind, i_ref2, j_hyp2 in ops:
            if kind in ("match", "sub"):
                ref_i = i_ref2
                wi = ref_index_to_word_idx(ref_i)
                if wi is not None and j_hyp2 is not None:
                    hyp_by_word[wi].append(hyp_phones_all[j_hyp2])
            elif kind == "del":
                ref_i = i_ref2
            elif kind == "ins":
                wi = ref_index_to_word_idx(ref_i)
                if wi is not None and j_hyp2 is not None:
                    hyp_by_word[wi].append(hyp_phones_all[j_hyp2])

        for wi in range(len(words)):
            hyp_by_word[wi] = filter_non_phonemic(hyp_by_word[wi])
            hyp_by_word[wi] = normalize_phone_sequence(hyp_by_word[wi])

        # ---------- (GOP) alinhamento CTC nos phones REF ----------
        ref_ids = [self.tok2id[p] for p in ref_phones_all if p in self.tok2id]
        segs = ctc_force_align(log_post, ref_ids, self.blank_id) if ref_ids else []

        # GOP-LR + detalhes por fonema (otimizado por lote de rivais)
        gop_lr_by_ref: List[Optional[float]] = [None] * len(ref_phones_all)
        phone_details_by_ref: List[Optional[dict]] = [None] * len(ref_phones_all)

        if segs:
            frame_spans: List[Tuple[int,int]] = []
            for seg in segs:
                t1e, t2e = _enforce_min_frames(seg.t1, seg.t2, T, min_frames=self.min_frames_per_phone)
                frame_spans.append((t1e, t2e))

            # Pré-calcula rivais max por segmento (all/vowel/cons)
            for idx, (t1e, t2e) in enumerate(frame_spans):
                if idx >= len(ref_phones_all):
                    break
                ph = ref_phones_all[idx]
                li = self.tok2id.get(ph, None)
                if li is None:
                    continue

                if t1e < 0 or t2e < 0 or t2e < t1e:
                    phone_details_by_ref[idx] = {
                        "phone": ph, "gop": float("-inf"), "threshold": float(self._thresh_for(ph)),
                        "pass": False, "t1": int(t1e), "t2": int(t2e),
                        "start_s": -1.0, "end_s": -1.0
                    }
                    gop_lr_by_ref[idx] = float("-inf")
                    continue

                seg_lp = log_post[t1e:t2e+1, :]  # [L,V]
                # mask_all = self.rival_mask_all
                # cls = is_vowel(ph)
                # mask_v = self.rival_mask_vowel if self.gop_rival_scope == "same_class" else None
                # mask_c = self.rival_mask_cons  if self.gop_rival_scope == "same_class" else None

                # rival_max_all = np.max(seg_lp[:, mask_all], axis=1) if mask_all is not None else None
                # rival_max_vowel = np.max(seg_lp[:, mask_v], axis=1) if mask_v is not None and mask_v.any() else None
                # rival_max_cons  = np.max(seg_lp[:, mask_c], axis=1) if mask_c is not None and mask_c.any() else None
                mask = self._rival_mask_for_phone(ph).copy()
                if 0 <= li < mask.size:
                    mask[li] = False  # EXCLUI o próprio alvo dos rivais
                # gval = gop_lr_with_mask_premax(seg_lp, li, rival_max_all, rival_max_vowel, rival_max_cons, cls)
                gval = gop_lr_excluding_self(seg_lp, li, mask)
                thr = self._thresh_for(ph)
                start_s = self._frame_to_time(T, dur_s_full, max(0, t1e))
                end_s   = self._frame_to_time(T, dur_s_full, max(0, t2e))

                gop_lr_by_ref[idx] = float(gval)
                phone_details_by_ref[idx] = {
                    "phone": ph,
                    "gop": float(gval),
                    "threshold": float(thr),
                    "pass": bool(gval >= thr),
                    "t1": int(t1e), "t2": int(t2e),
                    "start_s": float(start_s), "end_s": float(end_s),
                }

        # ---------- construir objetos por palavra ----------
        words_out: List[Dict] = []
        for wi, word in enumerate(words):
            target_phones = ref_phones_by_word[wi]
            produced_phones = hyp_by_word[wi]

            S_w, D_w, I_w, N_w = levenshtein_ops(target_phones, produced_phones)
            per_w = (S_w + D_w + I_w) / max(1, N_w)
            score_per = 1.0 - per_w

            a, b = ref_word_spans[wi]
            phone_infos = []
            weight_sum = 0.0
            pass_weight_sum = 0.0
            total_frames_word = 0
            for k in range(a, b+1):
                info = phone_details_by_ref[k] if k < len(phone_details_by_ref) else None
                if info is None: continue
                dur_frames = max(0, info["t2"] - info["t1"] + 1)
                total_frames_word += dur_frames
                w = float(dur_frames)
                weight_sum += w
                if info["pass"]: pass_weight_sum += w
                phone_infos.append(info)

            score_gop_weighted = (pass_weight_sum / weight_sum) if weight_sum > 0.0 else 0.0

            conf = confusion_list(target_phones, produced_phones)
            ops_summary = summarize_ops(target_phones, produced_phones)

            syllables = sum(1 for ph in target_phones if is_vowel(ph) is True)

            phones_gop = [
                {"phone": inf["phone"], "gop": float(inf["gop"]), "pass": bool(inf["pass"]), "threshold": float(inf["threshold"])}
                for inf in phone_infos
            ]

            words_out.append({
                "target_word": word,
                "target_phones": target_phones,
                "produced_phone": produced_phones,
                "word_score_per": float(score_per),
                "word_score_gop": float(score_gop_weighted),
                "phones": phone_infos,
                "phones_gop": phones_gop,
                "confusion": conf,
                "per_ops": ops_summary,
                "length_phones": int(len(target_phones)),
                "syllables": int(max(1, syllables)),
                "duration_frames": int(total_frames_word),
            })

        # ---------- métricas globais (PER, intelligibility) ----------
        S_g, D_g, I_g, N_g = levenshtein_ops(ref_phones_all, hyp_phones_all)
        per_sent = (S_g + D_g + I_g) / max(1, N_g)
        intelligibility = 1.0 - per_sent

        # ---------- Fluência ----------
        fluency_basic, fluency_details = compute_fluency(
            audio, log_post, self.tok2id, words_out, ref_phones_all,
            pred_ids_full, self.blank_id, trim_silence=True
        )

        # ---------- AER ----------
        ops_global = ops
        aer_mism = 0.0
        aer_total = 0.0
        per_feat_totals: Dict[str, Tuple[float,float]] = {}

        for kind, i_ref2, j_hyp2 in ops_global:
            ref_parts = _aer_expand_parts(ref_phones_all[i_ref2]) if i_ref2 is not None else []
            hyp_parts = _aer_expand_parts(hyp_phones_all[j_hyp2]) if j_hyp2 is not None else []

            if kind in ("match", "sub"):
                mism_w, total_w, per_feat, _ = articulatory_distance_weighted_lists(ref_parts, hyp_parts)
                aer_mism  += mism_w
                aer_total += total_w
                for k,(mw,tw) in per_feat.items():
                    pmw, ptw = per_feat_totals.get(k, (0.0,0.0))
                    per_feat_totals[k] = (pmw + mw, ptw + tw)

                # Lmin = min(len(ref_parts), len(hyp_parts))
                # deleções/inserções parciais já foram tratadas em articulatory_distance_weighted_lists()
                # aqui não precisamos somar por classe incrementalmente

            elif kind == "del":
                # todas as features do ref contam como mismatches
                for rph in ref_parts:
                    F1, keys1 = _feat_vec_cached(rph)
                    for k in keys1:
                        w = _feature_weight(k)
                        aer_mism += w; aer_total += w
                        pmw, ptw = per_feat_totals.get(k, (0.0,0.0))
                        per_feat_totals[k] = (pmw + w, ptw + w)

            elif kind == "ins":
                for hph in hyp_parts:
                    F2, keys2 = _feat_vec_cached(hph)
                    for k in keys2:
                        w = _feature_weight(k)
                        aer_mism += w; aer_total += w
                        pmw, ptw = per_feat_totals.get(k, (0.0,0.0))
                        per_feat_totals[k] = (pmw + w, ptw + w)

        aer = (aer_mism / aer_total) if aer_total > 0 else 0.0

        aer_by_feature = {k: {"mism": float(mw), "total": float(tw), "aer": (float(mw)/float(tw) if tw>0 else 0.0)}
                          for k, (mw,tw) in per_feat_totals.items()}

        # ---- AER por classe CONSISTENTE com os traços (correção) ----
        VOWEL_FEATS = {"h","b","round","rh"}
        CONS_FEATS  = {"voi","man","pla","lat","asp"}

        v_mism = sum(aer_by_feature[k]["mism"]  for k in VOWEL_FEATS if k in aer_by_feature)
        v_tot  = sum(aer_by_feature[k]["total"] for k in VOWEL_FEATS if k in aer_by_feature)
        c_mism = sum(aer_by_feature[k]["mism"]  for k in CONS_FEATS  if k in aer_by_feature)
        c_tot  = sum(aer_by_feature[k]["total"] for k in CONS_FEATS  if k in aer_by_feature)

        aer_by_class = {
            "vowel": {"mism": float(v_mism), "total": float(v_tot), "aer": (float(v_mism)/max(v_tot,1e-9))},
            "cons":  {"mism": float(c_mism), "total": float(c_tot), "aer": (float(c_mism)/max(c_tot,1e-9))},
        }

        # ---------- PER por classe e por posição (sem recontagem) ----------
        per_by_cls = stratified_per(ref_phones_all, hyp_phones_all)
        per_position = per_by_position_from_ops(ref_phones_all, ops, ref_word_spans)
        per_vowels = float(per_by_cls.get("vowel", {}).get("per", 0.0))
        per_cons   = float(per_by_cls.get("cons",  {}).get("per", 0.0))

        # ---------- IC por bootstrap ----------
        word_sdin = wordwise_sdin(ref_phones_by_word, hyp_by_word)
        per_ci = bootstrap_per_ci(word_sdin, n_boot=1000, seed=1337, alpha=0.05)

        # ---------- WAR variantes ----------
        def accept_exact(w):   return self._is_accept_exact(w["word_score_per"])
        def accept_tol(w):     return w["word_score_per"] >= self.war_tolerant_alpha
        def accept_hyb(w):     return (w["word_score_per"] >= self.war_hybrid_alpha) and (w["word_score_gop"] >= self.war_hybrid_beta)

        def rate(items, accept_fn):
            if not items: return 0.0
            return sum(1 for it in items if accept_fn(it)) / len(items)

        war_exact   = rate(words_out, accept_exact)
        war_tol     = rate(words_out, accept_tol)
        war_hybrid  = rate(words_out, accept_hyb)

        def war_by_len(items, accept_fn):
            buckets: Dict[int, Tuple[int,int]] = {}
            for it in items:
                L = int(it.get("length_phones", 0))
                ok = 1 if accept_fn(it) else 0
                if L not in buckets: buckets[L] = (0,0)
                a,b = buckets[L]
                buckets[L] = (a+ok, b+1)
            return {str(L): (a/b if b>0 else 0.0) for L,(a,b) in sorted(buckets.items())}

        war_len_exact  = war_by_len(words_out, accept_exact)
        war_len_tol    = war_by_len(words_out, accept_tol)
        war_len_hybrid = war_by_len(words_out, accept_hyb)

        def weighted_rate(items, weight_key, accept_fn):
            num = 0.0; den = 0.0
            for it in items:
                w = float(max(1, int(it.get(weight_key, 1))))
                den += w
                if accept_fn(it): num += w
            return (num / den) if den > 0 else 0.0

        war_w_duration  = weighted_rate(words_out, "duration_frames", accept_tol)
        war_w_syllables = weighted_rate(words_out, "syllables",       accept_tol)

        war_legacy = sum(1 for w in words_out if w["word_score_per"] >= self.word_ok_threshold) / max(1, len(words_out))

        return {
            "sentence_metrics": {
                "intelligibility": float(intelligibility),
                "word_accuracy_rate": float(war_legacy),

                "war_variants": {
                    "exact": float(war_exact),
                    "tolerant": {"alpha": float(self.war_tolerant_alpha), "value": float(war_tol)},
                    "hybrid":   {"alpha": float(self.war_hybrid_alpha),  "beta": float(self.war_hybrid_beta), "value": float(war_hybrid)},
                },
                "war_by_length": {
                    "exact":   war_len_exact,
                    "tolerant":war_len_tol,
                    "hybrid":  war_len_hybrid,
                },
                "war_weighted": {
                    "by_duration": float(war_w_duration),
                    "by_syllables": float(war_w_syllables),
                },

                "fluency": fluency_basic,
                "fluency_details": fluency_details,

                "aer": float(aer),
                "aer_overall": float(aer),
                "aer_by_feature": aer_by_feature,
                "aer_breakdown": {
                    "by_feature": aer_by_feature,
                    "by_class":   aer_by_class,
                },

                "per_vowels": float(per_vowels),
                "per_consonants": float(per_cons),

                "counts": {
                    "n_words": len(words),
                    "n_ref_phones": len(ref_phones_all),
                    "n_hyp_phones": len(hyp_phones_all),
                    "S": int(S_g), "D": int(D_g), "I": int(I_g),
                },
                "words": words_out,

                "per_components": { "S": int(S_g), "D": int(D_g), "I": int(I_g), "N_ref": int(N_g) },
                "per_by_class": per_by_cls,
                "per_by_position": per_position,
                "per_confidence_interval": per_ci,
            }
        }


# ========================= Singleton do serviço =========================

gop_thresholds = {
    # ================= VOGAIS MONOFTONGOS =================
    "i":  0.12,  # alta, anterior (estável)
    "ɪ":  0.10,
    "e":  0.10,  # se existir como monoftongo no vocabulário
    "ɛ":  0.10,
    "æ":  0.08,  # mais difícil p/ L2
    "ɑ":  0.12,
    "ʌ":  0.10,
    "ə":  0.08,  # schwa é instável (mais permissivo)
    "ɝ":  0.07,  # rótica é difícil — mais permissivo
    "ɔ":  0.09,
    "ʊ":  0.08,  # curta e confusável
    "u":  0.12,

    # ================= DITONGOS =================
    "eɪ": 0.09,
    "oʊ": 0.09,
    "aɪ": 0.08,
    "aʊ": 0.08,
    "ɔɪ": 0.08,

    # ================= STOPS =================
    "p": 0.10, "b": 0.10,
    "t": 0.10, "d": 0.10,
    "k": 0.10, "g": 0.10, "ɡ": 0.10,  # g/ɡ mapeiam para o mesmo

    # ================= AFRICADAS =================
    "tʃ": 0.08,
    "dʒ": 0.08, "ʤ": 0.08,  # normalização cobre ambas

    # ================= FRICATIVAS =================
    "f": 0.09, "v": 0.09,
    "θ": 0.07, "ð": 0.07,  # difíceis p/ L2 → mais permissivo
    "s": 0.10, "z": 0.10,
    "ʃ": 0.09, "ʒ": 0.08,
    "h": 0.08,

    # ================= NASAIS =================
    "m": 0.10, "n": 0.10, "ŋ": 0.10,

    # ================= LÍQUIDAS / APROXIMANTES =================
    "l": 0.09,
    "ɹ": 0.07,  # /r/ americano é difícil
    "w": 0.09,
    "j": 0.09,

    # (tokens não-fonêmicos como "sil"/"spn"/"nsn" não recebem threshold)
}
gop_thresholds = {
    # ================= VOGAIS MONOFTONGOS =================
    "i":  0.12,  # alta, anterior (estável)
    "ɪ":  0.10,
    "e":  0.10,  # se existir como monoftongo no vocabulário
    "ɛ":  0.10,
    "æ":  0.08,  # mais difícil p/ L2
    "ɑ":  0.12,
    "ʌ":  0.10,
    "ə":  0.08,  # schwa é instável (mais permissivo)
    "ɝ":  0.07,  # rótica é difícil — mais permissivo
    "ɔ":  0.09,
    "ʊ":  0.08,  # curta e confusável
    "u":  0.12,

    # ================= DITONGOS =================
    "eɪ": 0.09,
    "oʊ": 0.09,
    "aɪ": 0.08,
    "aʊ": 0.08,
    "ɔɪ": 0.08,

    # ================= STOPS =================
    "p": 0.10, "b": 0.10,
    "t": 0.10, "d": 0.10,
    "k": 0.10, "g": 0.10, "ɡ": 0.10,  # g/ɡ mapeiam para o mesmo

    # ================= AFRICADAS =================
    "tʃ": 0.08,
    "dʒ": 0.08, "ʤ": 0.08,  # normalização cobre ambas

    # ================= FRICATIVAS =================
    "f": 0.09, "v": 0.09,
    "θ": 0.07, "ð": 0.07,  # difíceis p/ L2 → mais permissivo
    "s": 0.10, "z": 0.10,
    "ʃ": 0.09, "ʒ": 0.08,
    "h": 0.08,

    # ================= NASAIS =================
    "m": 0.10, "n": 0.10, "ŋ": 0.10,

    # ================= LÍQUIDAS / APROXIMANTES =================
    "l": 0.09,
    "ɹ": 0.07,  # /r/ americano é difícil
    "w": 0.09,
    "j": 0.09,

    # (tokens não-fonêmicos como "sil"/"spn"/"nsn" não recebem threshold)
}

_SERVICE_SINGLETON: Optional[PronEvaluator] = None


def get_pron_service() -> PronEvaluator:
    global _SERVICE_SINGLETON
    if _SERVICE_SINGLETON is None:
        _SERVICE_SINGLETON = PronEvaluator(
            # gop_thresholds=gop_thresholds,
            global_gop_threshold=0.4,
            gop_rival_scope="same_class",
            word_ok_threshold=0.8,
            war_tolerant_alpha=0.8,
            war_hybrid_alpha=0.8,
            war_hybrid_beta=0.2,
        )
    return _SERVICE_SINGLETON

# Perfis sugeridos (ajuste à vontade)
PROFILES = {
    "iniciante":     {"alpha": 0.75, "beta": 0.20},
    "intermediario": {"alpha": 0.80, "beta": 0.40},
    "estrito":       {"alpha": 0.88, "beta": 0.70},
}


# EXEMPLO DE USO:
# from pprint import pprint
# words = result["sentence_metrics"]["words"]   # do seu dicionário atual

from typing import Dict, Any, List

def score_pronunciation(sample: Dict[str, Any], alpha: float = 0.80, beta: float = 0.40) -> Dict[str, Any]:
    """
    Decide acerto/erro por palavra combinando:
      - PER estrutural: word_score_per ≥ alpha
      - Qualidade (GOP): word_score_gop ≥ beta  E todos os fones 'pass' (quando disponível)
    Também calcula PER global a partir de S/D/I do topo do JSON.

    Parâmetros:
        sample: dicionário no formato do seu retorno.
        alpha: limiar PER por palavra (>= para passar). 0.80 é um bom ponto de partida para apps educacionais.
        beta:  limiar GOP por palavra (>= para passar). 0.40–0.60 são comuns quando GOP é normalizado em [0,1].

    Retorna:
        {
          'alpha': float, 'beta': float,
          'overall': {'per': float, 'passed_rate': float},
          'words': [
             {
               'target': str,
               'per': float, 'gop': float,
               'per_pass': bool, 'gop_pass': bool, 'hybrid_pass': bool,
               'issues': {'structure': [...], 'quality': [...]}
             }, ...
          ],
          'phoneme_todo': [{'word': str, 'phone': str, 'reason': str, 'gop': float, 't_sec': [start, end]}, ...]
        }
    """
    # --- PER global a partir dos componentes do topo (se existirem) ---
    per_global = None
    if 'per_components' in sample:
        pc = sample['per_components']
        S, D, I = pc.get('S', 0), pc.get('D', 0), pc.get('I', 0)
        N = pc.get('N_ref', pc.get('N', None))
        if N and N > 0:
            per_global = (S + D + I) / N

    words_out: List[Dict[str, Any]] = []
    phoneme_todo: List[Dict[str, Any]] = []

    for w in sample.get('words', []):
        target = w.get('target_word')
        per = float(w.get('word_score_per', 0.0))
        gop_word = float(w.get('word_score_gop', 0.0))

        # Checagem de qualidade por fone (quando houver campo 'phones' com 'pass')
        phones = w.get('phones', [])
        all_phones_pass = True
        quality_issues = []
        for p in phones:
            # Se existir a flag 'pass', usamos; se não existir, aceitamos o fone
            p_pass = p.get('pass', True)
            if not p_pass:
                all_phones_pass = False
                quality_issues.append(
                    f"/{p.get('phone')}/ abaixo do limiar (gop={p.get('gop'):.3f} < thr={p.get('threshold')})"
                    if 'gop' in p and 'threshold' in p else f"/{p.get('phone')}/ abaixo do limiar"
                )
                # Guardar para uma lista agregada de “o que treinar”
                t1, t2 = p.get('start_s'), p.get('end_s')
                phoneme_todo.append({
                    'word': target, 'phone': p.get('phone'), 'reason': 'gop_low',
                    'gop': float(p.get('gop', 0.0)), 't_sec': [t1, t2]
                })

        # Erros estruturais a partir do alinhamento (quando vier em 'confusion'/'per_ops')
        structure_issues = []
        for op in (w.get('confusion') or []):
            if op.get('op', '').startswith('sub'):
                structure_issues.append(f"substituição: {op.get('ref')}→{op.get('hyp')}")
            elif op.get('op') == 'del':
                structure_issues.append(f"deleção: {op.get('ref')}")
            elif op.get('op') == 'ins':
                structure_issues.append(f"inserção: {op.get('hyp')}")

        # Decisão por palavra
        per_pass = per >= alpha
        gop_pass = (gop_word >= beta) and all_phones_pass
        hybrid_pass = per_pass and gop_pass

        words_out.append({
            'target': target,
            'per': per,
            'gop': gop_word,
            'per_pass': per_pass,
            'gop_pass': gop_pass,
            'hybrid_pass': hybrid_pass,
            'issues': {'structure': structure_issues, 'quality': quality_issues},
        })

    passed_rate = sum(1 for w in words_out if w['hybrid_pass']) / max(len(words_out), 1)

    return {
        'alpha': alpha, 'beta': beta,
        'overall': {'per': per_global, 'passed_rate': passed_rate},
        'words': words_out,
        'phoneme_todo': phoneme_todo
    }



def gerar_feedback_pronuncia(
    data: Dict[str, Any],
    alpha: float = 0.80,  # limiar de acerto por alinhamento (PER) para dizer "falou a palavra certa"
    beta: float = 0.40,   # limiar de qualidade acústica (GOP) para dizer "soou natural/estável"
    max_palavras: int = 10 # máximo de palavras com feedback detalhado
) -> Dict[str, Any]:
    """
    Converte as métricas do seu analisador em um feedback amigável para o usuário final.
    Entrada: dicionário com os campos do seu JSON (intelligibility, words, aer_breakdown, etc.)
    Saída: dicionário pronto para renderizar na UI do app.
    """

    # --------- Helpers ---------
    def status_word(per: float, gop: float) -> Tuple[str, str]:
        """
        Classifica a palavra combinando PER (alinhamento) e GOP (qualidade por fone).
        """
        if per >= alpha and gop >= beta:
            return ("correta", "Pronúncia correta.")
        if per >= alpha and gop < beta:
            return ("soou estranha", "Sequência correta, mas o som poderia ficar mais estável/natural.")
        if per < alpha and gop >= beta:
            return ("fone trocado/omitido", "O som foi claro, mas houve troca/omissão na sequência de sons.")
        return ("precisa de ajuste", "Houve trocas/omissões e a qualidade ainda pode melhorar.")

    def tip_for_feature(code: str) -> str:
        """
        Mensagens curtas por traço articulatório.
        """
        tips = {
            "h":   "Altura da vogal: feche mais (língua alta) para /i, u/; abra mais (língua baixa) para /æ, ɑ/.",
            "b":   "Anterioridade da vogal: empurre a língua para frente em /i, ɪ/ e para trás em /u, oʊ/.",
            "round": "Arredondamento: arredonde mais os lábios em /u, oʊ/; relaxe em /i, ɪ/.",
            "rh":  "Roticidade: enrole levemente a língua e mantenha o r-coloring em /ɝ, ɚ/ (ex.: bird).",
            "pla": "Ponto de articulação: atenção ao lugar da consoante (ex.: /k/ é mais atrás que /t/).",
            "man": "Modo de articulação: mantenha fricativas com fluxo de ar contínuo (ex.: /f, s, ʃ/) e oclusivas bem fechadas/soltas (ex.: /p, t, k/).",
            "asp": "Aspiração: dê um sopro curto após /p, t, k/ em início de sílaba (ex.: 'pin', 'top', 'cat').",
            "lat": "Lateralidade: no /l/, deixe o ar passar pelas laterais da língua.",
            "voi": "Vozeamento: ative as pregas vocais nas sonoras (/b, d, g, v, z/); desligue nas surdas (/p, t, k, f, s/).",
        }
        return tips.get(code, "Ajuste de articulação necessário nesse traço.")

    def human_op(op: str) -> str:
        """
        Converte operações PER em linguagem simples.
        """
        if op.startswith("sub("):
            # sub(k→t)
            troca = op[4:-1]
            return f"troca de som ({troca})"
        if op.startswith("del("):
            # del(ɪ)
            faltou = op[4:-1]
            return f"faltou o som /{faltou}/"
        if op.startswith("ins("):
            extra = op[4:-1]
            return f"sobrou o som /{extra}/"
        return "ok"

    # --------- Resumo geral ---------
    intelligibility = float(data.get("intelligibility", 0))
    war = float(data.get("word_accuracy_rate", 0))
    dur_s = data.get("fluency", {}).get("duration_s", None)
    wps = data.get("fluency", {}).get("words_per_sec", None)
    pause_ratio = data.get("fluency", {}).get("pause_ratio", None)
    fluency_level = data.get("fluency", {}).get("level", None)

    resumo_texto = []
    resumo_texto.append(f"Inteligibilidade: {intelligibility:.2f} (0–1).")
    resumo_texto.append(f"Acurácia de palavras: {war:.2f}.")
    if dur_s is not None:
        resumo_texto.append(f"Duração: {dur_s:.2f}s.")
    if wps is not None:
        resumo_texto.append(f"Velocidade: {wps:.2f} palavras/s.")
    if pause_ratio is not None:
        resumo_texto.append(f"Proporção de pausas: {pause_ratio:.2f}.")
    if fluency_level:
        resumo_texto.append(f"Fluência: {fluency_level}.")

    # --------- Top traços com maior erro (AER) ---------
    features = (data.get("aer_breakdown") or data.get("aer_by_feature") or {}).get("by_feature", data.get("aer_by_feature", {}))
    feature_list = []
    for code, stats in features.items():
        try:
            aer = float(stats.get("aer", 0.0))
            mism = float(stats.get("mism", 0.0))
            total = float(stats.get("total", 0.0))
        except Exception:
            aer, mism, total = 0.0, 0.0, 0.0
        feature_list.append((code, aer, mism, total))
    # Ordena por AER decrescente
    feature_list.sort(key=lambda x: x[1], reverse=True)
    # Seleciona os principais acima de 0.15 (ajuste livre)
    destaques = [f for f in feature_list if f[1] >= 0.15][:3]

    pontos_de_ajuste = []
    for code, aer, mism, total in destaques:
        pontos_de_ajuste.append({
            "traco": code,
            "aer": round(aer, 3),
            "explicacao": tip_for_feature(code)
        })

    # --------- Feedback por palavra ---------
    palavras_raw = data.get("words", [])
    feedback_palavras = []
    for w in palavras_raw[:max_palavras]:
        alvo = w.get("target_word")
        per = float(w.get("word_score_per", 0.0))
        gop = float(w.get("word_score_gop", 0.0))
        status, msg = status_word(per, gop)

        # Explica rapidamente o que houve (substituição/omissão/etc.)
        ops = w.get("per_ops", []) or []
        problemas = [human_op(op) for op in ops if human_op(op) != "ok"]
        problemas = list(dict.fromkeys(problemas))  # dedup
        detalhe = None
        if problemas:
            detalhe = "; ".join(problemas)

        # Sugestão específica se houver substituição que envolva lugar/modo
        sugestao_curta = None
        if any("troca de som" in p for p in problemas):
            sugestao_curta = "Atenção ao ponto/movimento da língua para diferenciar esses sons."

        if any("faltou o som" in p for p in problemas):
            sugestao_curta = (sugestao_curta + " " if sugestao_curta else "") + "Garanta que todos os sons da palavra sejam realizados."

        feedback_palavras.append({
            "palavra": alvo,
            "status": status,
            "mensagem": msg,
            "per": round(per, 3),
            "gop": round(gop, 3),
            "detalhes_alinhamento": detalhe,
            "sugestao": sugestao_curta
        })

    # --------- Sinal geral (para pintar uma badge na UI) ---------
    # Regras simples: bom ≥0.85, ok 0.70–0.84, atenção <0.70 (use algo que faça sentido no seu público)
    if war >= 0.85 and intelligibility >= 0.90:
        sinal = "excelente"
    elif war >= 0.70 and intelligibility >= 0.80:
        sinal = "bom"
    else:
        sinal = "precisa de prática"

    # --------- Dica geral por tamanho de palavra (opcional) ---------
    war_by_len = (data.get("war_by_length") or {}).get("hybrid") or (data.get("war_by_length") or {}).get("exact") or {}
    dica_tamanho = None
    if war_by_len:
        # pega a pior faixa (menor valor)
        try:
            pior_len = min(war_by_len.items(), key=lambda kv: kv[1])
            dica_tamanho = f"Palavras com {pior_len[0]} fones tiveram mais dificuldade (acerto {pior_len[1]:.2f})."
        except Exception:
            pass

    # --------- Monta saída ---------
    return {
        "sinal_geral": sinal,
        "resumo": " ".join(resumo_texto),
        "pontos_de_ajuste": pontos_de_ajuste,          # até 3 traços prioritários com dica curta
        "feedback_por_palavra": feedback_palavras,      # até max_palavras itens
        "parametros_usados": {
            "alpha_per": alpha,
            "beta_gop": beta
        },
        "observacoes": [
            "Uma palavra é considerada correta quando word_score_per ≥ α e word_score_gop ≥ β.",
            "Se PER estiver alto mas GOP baixo: a sequência está certa, mas os sons podem soar tensos/instáveis.",
            "Se GOP estiver alto mas PER baixo: os sons são claros, porém houve troca/omissão na sequência.",
            *(f"{dica_tamanho}" if dica_tamanho else "")
        ]
    }


# ========================= Execução direta =========================

from typing import List, Dict, Any



def weighted_word_accuracy(words: List[Dict[str, Any]],
                           weights: Dict[str, float] = {"perfect": 1.0, "high": 0.80, "medium": 0.5, "low":0.0}) -> float:

    if not words:
        return 0.0
    score_sum = 0.0
    for w in words:
        lvl = (w.get("pronunciation_quality") or "").lower()
        score_sum += weights.get(lvl, 0.0)  # desconhecidos contam como 0.0
    return score_sum / len(words)

def formater_output(data,
    weight_per: float = 0.6,
    weight_gop: float = 0.4,
    ops_penalty: float = 0.05,
                    ):
    # t_medium, t_low = (0.70, 0.85)
    thresholds4 = (0.90, 0.80, 0.70)
    intelligibility = float(data.get("intelligibility", 0))
    # war = float(data.get("word_accuracy_rate", 0))
    fluency_level = data.get("fluency", {}).get("level", None)

    words_raw = data.get("words", [])
    words = []
    for w in words_raw:
        per = float(w.get("word_score_per", 0.0))
        gop = float(w.get("word_score_gop", 0.0))
        ops =  list(filter(lambda op: op != "match", w.get("per_ops", [])))

        if len(ops) == 0 and per == 1.0:
            gop = 1.0

        score = weight_per * per + weight_gop * gop

        # penalty for deletions/insertions (more critical for intelligibility)
        n_penalized_ops = sum(1 for op in ops if op.startswith("del(") or op.startswith("ins("))
        score -= min(n_penalized_ops * ops_penalty, 0.15)

        # clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        t_perfect, t_high, t_medium = thresholds4

        if score >= t_perfect:
            cls = "perfect"
        elif score >= t_high:
            cls = "high"
        elif score >= t_medium:
            cls = "medium"
        else:
            cls = "low"

        words.append({
            "target_word": w.get("target_word"),
            "target_phones": w.get("target_phones",[]),
            "produced_phone": w.get("produced_phone",[]),
            "per": per,
            "gop": gop,
            "ops": ops,
            "pronunciation_quality":cls,
        })


    return {
        "intelligibility": intelligibility,
        "word_accuracy_rate": weighted_word_accuracy(words),
        "fluency_level": fluency_level,
        "words": words,
    }


if __name__ == "__main__":
    service = get_pron_service()
    # "meu_audio.wav" "audio.mp3"
    result = service.evaluate("audio.mp3", "She looked guilty after lying to her friend")

    data = formater_output(result["sentence_metrics"])
    print(data)
