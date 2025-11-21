# Análise de Segurança: api.py e Dependências

Esta análise identifica vulnerabilidades de segurança no arquivo `api.py` e suas dependências, classificando-as por severidade e fornecendo recomendações de correção.

## Resumo Executivo

O arquivo `api.py` expõe uma API FastAPI sem mecanismos de autenticação ou autorização, tornando-a vulnerável a acesso não autorizado se exposta em rede. Foi identificada uma vulnerabilidade de **Path Traversal** que pode permitir leitura de arquivos arbitrários (limitado a arquivos .mp3). A configuração de CORS é excessivamente permissiva, e o servidor de desenvolvimento está configurado para escutar em todas as interfaces de rede (`0.0.0.0`).

## Vulnerabilidades Identificadas

### 1. Path Traversal (Alta Severidade)
**Localização:** `api.py`, endpoint `GET /audio/{audio_id}` (função nomeada incorretamente como `delete_card` na linha 280).
**Descrição:** O parâmetro `audio_id` é concatenado diretamente ao caminho do diretório:
```python
b64 = mp3_to_base64(str("temp_files/" + audio_id))
```
Um atacante pode fornecer um `audio_id` contendo sequências como `../../` para acessar arquivos fora do diretório `temp_files`. Embora a função `mp3_to_base64` verifique a extensão `.mp3`, isso ainda permite a exfiltração de qualquer arquivo MP3 no sistema de arquivos do servidor.
**Recomendação:** Validar se `audio_id` contém apenas caracteres seguros (alfanuméricos, hífens, underscores) ou usar `os.path.basename()` para garantir que o acesso seja restrito ao diretório `temp_files`.

### 2. Ausência de Autenticação e Autorização (Alta Severidade)
**Localização:** Todo o arquivo `api.py`.
**Descrição:** Não há middleware ou dependências de segurança configuradas. Qualquer pessoa com acesso à rede pode chamar qualquer endpoint, incluindo criar, modificar e deletar decks e cards, além de usar a API de chat (que consome créditos da OpenAI).
**Recomendação:** Implementar autenticação (ex: OAuth2 com JWT ou API Keys). Adicionar dependências de segurança (`Depends(...)`) em todos os endpoints sensíveis.

### 3. Configuração CORS Permissiva (Média Severidade)
**Localização:** `api.py`, linhas 169-175.
**Descrição:**
```python
allow_origins=["*"]
```
Isso permite que qualquer site faça requisições para sua API via navegador. Se a API estiver rodando localmente e você visitar um site malicioso, esse site pode interagir com sua API (CSRF/CORS attacks).
**Recomendação:** Restringir `allow_origins` para domínios específicos (ex: `http://localhost:3000` se for um frontend React local).

### 4. Exposição de Rede em Modo de Desenvolvimento (Média Severidade)
**Localização:** `api.py`, linha 389.
**Descrição:**
```python
uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
```
O host `0.0.0.0` expõe o serviço para todas as interfaces de rede. Combinado com a falta de autenticação, qualquer dispositivo na mesma rede Wi-Fi/LAN pode acessar a API.
**Recomendação:** Alterar para `host="127.0.0.1"` para restringir o acesso apenas à máquina local, a menos que o acesso externo seja explicitamente necessário (e nesse caso, autenticação é mandatória).

### 5. Validação de Entrada Insuficiente (Baixa Severidade)
**Localização:** `api.py`, endpoint `POST /chat-stream`.
**Descrição:** O endpoint aceita um `dict` genérico (`payload: dict = Body(...)`) e extrai campos manualmente. Isso impede que o FastAPI gere documentação correta (Swagger UI) e validação automática de tipos.
**Recomendação:** Criar um modelo Pydantic (ex: `ChatRequest`) para definir a estrutura esperada do corpo da requisição.

### 6. Risco de Prompt Injection (Geral)
**Localização:** `app/services/chat_service.py`.
**Descrição:** O input do usuário é concatenado diretamente ao histórico de mensagens enviado para a OpenAI. Embora seja o funcionamento padrão de LLMs, usuários maliciosos podem tentar manipular as instruções do sistema.
**Recomendação:** Monitorar o uso e considerar camadas de validação de input antes de enviar ao modelo, se o sistema for exposto publicamente.

## Análise de Dependências

O arquivo `pyproject.toml` define as dependências.
*   **Pontos Positivos:** As versões estão "pinadas" (ex: `>=2.12.3`), o que ajuda na estabilidade.
*   **Atenção:**
    *   `uvicorn` com `reload=True` em produção é inseguro (consumo de recursos e exposição de código).
    *   Certifique-se de manter `fastapi`, `pydantic` e `uvicorn` atualizados para receber correções de segurança.

## Plano de Ação Sugerido

1.  **Corrigir Path Traversal:** Alterar a lógica de `audio_id` imediatamente.
2.  **Restringir Rede:** Mudar o host para `127.0.0.1`.
3.  **Implementar Autenticação:** Adicionar uma camada básica de proteção (mesmo que seja uma API Key simples no header) se planeja expor o serviço.
4.  **Refinar CORS:** Listar apenas as origens necessárias.
