import os
import re
import wave
import shutil
import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

class AudioServiceError(RuntimeError):
    pass

class AudioService:
    def __init__(self, target_sr: int = 16000, mono: bool = True, apply_fade_ms: int = 3):
        self.target_sr = target_sr
        self.mono = mono
        self.apply_fade_ms = apply_fade_ms

    @staticmethod
    def sanitize_filename(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"[^\w\-_\.]", "_", s, flags=re.UNICODE)
        return s[:40] or "word"

    def convert_to_wav(self, input_path: str, output_path: Optional[str] = None, overwrite: bool = True) -> str:
        if not os.path.isfile(input_path):
            raise AudioServiceError(f"Arquivo de entrada não encontrado: {input_path}")
        if shutil.which("ffmpeg") is None:
            raise AudioServiceError("ffmpeg não encontrado no PATH.")

        if output_path is None:
            base, _ = os.path.splitext(input_path)
            output_path = base + ".wav"

        if os.path.exists(output_path) and not overwrite:
            raise AudioServiceError(f"Arquivo de saída já existe: {output_path}")

        cmd = ["ffmpeg", "-y" if overwrite else "-n", "-i", input_path, "-vn",
               "-acodec", "pcm_s16le", "-ar", str(self.target_sr)]
        if self.mono:
            cmd += ["-ac", "1"]
        cmd.append(output_path)

        logger.info("Convertendo para WAV: %s -> %s", input_path, output_path)
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise AudioServiceError(e.stderr.decode(errors="ignore")) from e

        self._validate_wav(output_path)
        return output_path

    def _validate_wav(self, path: str) -> None:
        try:
            with wave.open(path, "rb") as wf:
                if wf.getframerate() != self.target_sr:
                    raise AudioServiceError("SR diferente do esperado.")
                if self.mono and wf.getnchannels() != 1:
                    raise AudioServiceError("Canal diferente de mono.")
                if wf.getsampwidth() != 2:
                    raise AudioServiceError("Profundidade diferente de 16-bit.")
        except Exception as e:
            raise AudioServiceError(f"Falha ao validar WAV: {e}") from e

    def cut_precise(
            self,
            input_file: str,
            start: float,
            end: float,
            out_file: str,
            *,
            mp3_mode: str = "vbr",  # "vbr" (qualidade) ou "cbr" (bitrate)
            mp3_quality: int = 2,  # 0=melhor, 9=pior (VBR -q:a)
            mp3_bitrate: str = "192k",  # CBR -b:a (ignore se mp3_mode="vbr")
    ) -> None:
        dur = max(0.0, end - start)
        if dur <= 0:
            logger.warning("Intervalo inválido: %.6f–%.6f", start, end)
            return

        # Filtro: corte preciso + (opcional) micro-fade
        if self.apply_fade_ms and dur > (2 * self.apply_fade_ms / 1000.0):
            fi = self.apply_fade_ms / 1000.0
            fo = self.apply_fade_ms / 1000.0
            filter_expr = (
                f"atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:st=0:d={fi:.6f},afade=t=out:st={dur - fo:.6f}:d={fo:.6f}"
            )
        else:
            filter_expr = f"atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS"

        ext = os.path.splitext(out_file)[1].lower()

        # Codificação de saída conforme extensão
        if ext == ".mp3":
            # MP3 com libmp3lame. Mantemos mono e SR alvo (pode ser 16k).
            cmd = [
                "ffmpeg", "-y",
                "-i", input_file,
                "-filter_complex", filter_expr,
                "-ac", "1",
                "-ar", str(self.target_sr),
                "-c:a", "libmp3lame",
            ]
            if mp3_mode.lower() == "cbr":
                cmd += ["-b:a", mp3_bitrate]  # ex.: "192k"
            else:
                cmd += ["-q:a", str(mp3_quality)]  # ex.: 2 (VBR alta qualidade)
            cmd += [out_file]
        else:
            # WAV PCM 16-bit (comportamento original)
            cmd = [
                "ffmpeg", "-y",
                "-i", input_file,
                "-filter_complex", filter_expr,
                "-c:a", "pcm_s16le",
                "-ar", str(self.target_sr),
                "-ac", "1",
                out_file
            ]

        logger.debug("Cortando: %s [%.3f–%.3f] -> %s", input_file, start, end, out_file)
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise AudioServiceError(e.stderr.decode(errors="ignore")) from e

    def cut_precise_to_bytes(
            self,
            input_file: str,
            start: float,
            end: float,
            *,
            fmt: str = "mp3",
            mp3_mode: str = "vbr",
            mp3_quality: int = 2,
            mp3_bitrate: str = "192k",
    ) -> bytes:
        """
        Variante em memória do corte preciso. Retorna os bytes do trecho
        solicitado para evitar gravações intermediárias em disco.
        """
        dur = max(0.0, end - start)
        if dur <= 0:
            logger.warning("Intervalo inválido: %.6f–%.6f", start, end)
            return b""

        if self.apply_fade_ms and dur > (2 * self.apply_fade_ms / 1000.0):
            fi = self.apply_fade_ms / 1000.0
            fo = self.apply_fade_ms / 1000.0
            filter_expr = (
                f"atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:st=0:d={fi:.6f},afade=t=out:st={dur - fo:.6f}:d={fo:.6f}"
            )
        else:
            filter_expr = f"atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS"

        fmt = (fmt or "mp3").lower()
        base_cmd = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-filter_complex", filter_expr,
            "-ac", "1",
            "-ar", str(self.target_sr),
        ]

        if fmt == "mp3":
            cmd = base_cmd + ["-c:a", "libmp3lame"]
            if mp3_mode.lower() == "cbr":
                cmd += ["-b:a", mp3_bitrate]
            else:
                cmd += ["-q:a", str(mp3_quality)]
            cmd += ["-f", "mp3", "pipe:1"]
        elif fmt == "wav":
            cmd = base_cmd + [
                "-c:a", "pcm_s16le",
                "-f", "wav",
                "pipe:1",
            ]
        else:
            raise AudioServiceError(f"Formato não suportado para corte: {fmt}")

        logger.debug("Cortando (bytes): %s [%.3f–%.3f] -> %s", input_file, start, end, fmt)
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            raise AudioServiceError(e.stderr.decode(errors="ignore")) from e

        return result.stdout

    # def cut_precise(self, input_file: str, start: float, end: float, out_file: str) -> None:
    #     dur = max(0.0, end - start)
    #     if dur <= 0:
    #         logger.warning("Intervalo inválido: %.6f–%.6f", start, end)
    #         return
    #
    #     if self.apply_fade_ms and dur > (2 * self.apply_fade_ms / 1000.0):
    #         fi = self.apply_fade_ms / 1000.0
    #         fo = self.apply_fade_ms / 1000.0
    #         filter_expr = (
    #             f"atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS,"
    #             f"afade=t=in:st=0:d={fi:.6f},afade=t=out:st={dur-fo:.6f}:d={fo:.6f}"
    #         )
    #     else:
    #         filter_expr = f"atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS"
    #
    #     cmd = ["ffmpeg", "-y", "-i", input_file, "-filter_complex", filter_expr,
    #            "-c:a", "pcm_s16le", "-ar", str(self.target_sr), "-ac", "1", out_file]
    #
    #     logger.debug("Cortando: %s [%.3f–%.3f] -> %s", input_file, start, end, out_file)
    #     try:
    #         subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    #     except subprocess.CalledProcessError as e:
    #         raise AudioServiceError(e.stderr.decode(errors="ignore")) from e
