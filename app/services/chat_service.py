# main.py
import os
import json
import re
from openai import OpenAI

# Certifique-se de ter essas importações corretas no seu projeto
from app.mcp.mcp_client import call_generate_tts_audio_sync, call_pronunciation_practice_sync, call_create_new_card_sync
# Se tiver uma função para pronúncia, importe-a aqui também, ex:
# from app.mcp.mcp_client import call_pronunciation_practice_sync
from app.mcp.tools import TOOLS


class ChatService:
    def __init__(self, model_name="gpt-4o-mini"):  # Corrigido para um modelo válido
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in environment variables.")

        self.client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name

    def _extract_deck_id(self, history) -> str | None:
        """
        Extracts the deck ID from the history if present.
        Look for: "O usuário está praticando o DECK DE ID <uuid>"
        """
        pattern = r"O usuário está praticando o DECK DE ID ([a-f0-9\-]+)"
        for msg in history:
            content = msg.get("content", "")
            if content:
                match = re.search(pattern, content)
                if match:
                    return match.group(1)
        return None

    def generate_answer_stream(self, history, user_message: str):
        """
        history: lista de dicts {"role": "user"|"assistant", "content": str}
        user_message: mensagem atual do usuário

        Gera pedaços de texto (chunks) da resposta final para streaming.
        """
        
        # Tenta extrair o deck_id do histórico
        active_deck_id = self._extract_deck_id(history)

        # Monta histórico para a API
        # CORREÇÃO 1: Concatenação correta das strings no system prompt
        system_prompt = (
            "Você é um assistente útil e responde em português. "
            "IMPORTANTE: Não use a função 'generate_tts_audio' a menos que o usuário peça explicitamente para ouvir ou gerar áudio. "
            "Se o usuário pedir apenas exemplos de frases, apenas escreva o texto.\n"
            "Quando usar a função 'generate_tts_audio', ela retorna um JSON com 'file_path'. "
            "Sempre que usar essa função, termine sua resposta com uma linha EXATAMENTE neste formato:\n"
            "AUDIO_FILE: <file_path_do_áudio>\n"
            "Não coloque texto depois dessa linha.\n"
            "Quando usar a função: 'pronunciation_practice', quando o usuario quiser praticar a pronuncia "
            "de uma palavra ou frase em ingles, ela retorna um JSON com 'practice'. "
            "Sempre que usar essa função, termine sua resposta com uma linha EXATAMENTE neste formato:\n"
            "START_PRACTICE: <text>\n"
            "Não coloque texto depois dessa linha.\n"
            "Quando usar a função 'create_new_card', você DEVE fornecer um 'deck_id'. "
            "Se o usuário não especificou um deck, PERGUNTE em qual deck ele deseja criar o card antes de chamar a função. "
            "Não invente um deck_id. "
            "A função retorna um JSON. Se o JSON contiver 'error', informe o usuário sobre o erro. "
            "Se retornar os dados do card, confirme a criação com sucesso."
        )

        if active_deck_id:
            system_prompt += f"\n\nO usuário está praticando no DECK ID: {active_deck_id}. Use este ID para criar cards se solicitado."

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        

        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_message})


        # 1ª chamada: modelo decide se chama tool ou responde direto
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # =====================
        # CASO 1: modelo NÃO usa nenhuma tool
        # =====================
        if not message.tool_calls:
            # Fazemos uma chamada separada com stream=True para ir mandando os tokens
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

            return

        # =====================
        # CASO 2: modelo chamou uma ou mais tools
        # =====================

        # Adiciona a "intenção" de tool ao histórico
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": message.tool_calls,
            }
        )

        # Executa cada tool_call na mão (via MCP)
        for tool_call in message.tool_calls:
            if tool_call.type != "function":
                continue

            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            tool_result_content = ""

            # CORREÇÃO 2: Tratamento das diferentes tools
            if name == "generate_tts_audio":
                text = args.get("text")
                tts_result = call_generate_tts_audio_sync(text)
                # print("tts_result: ", tts_result) # Debug opcional
                tool_result_content = json.dumps(tts_result)

            elif name == "pronunciation_practice":
                target_text = args.get("text") or args.get("phrase")

                # Exemplo de mock (simulação) caso não tenha a função importada ainda:
                practice_result = call_pronunciation_practice_sync(target_text)
                # print(f"Chamando pronunciation_practice para: {target_text}")
                # practice_result = {"practice": "started", "text": target_text}

                tool_result_content = json.dumps(practice_result)

            elif name == "create_new_card":
                content = args.get("content")
                deck_id = args.get("deck_id")
                card_result = call_create_new_card_sync(content, deck_id)
                tool_result_content = json.dumps(card_result)
                print("tool_result_content: ", tool_result_content)

            else:
                # Caso o modelo alucine uma função que não existe
                tool_result_content = json.dumps({"error": "Function not found"})

            # manda o resultado da tool de volta pro modelo
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result_content,
                }
            )

        # 2ª chamada: agora o modelo vê o resultado da tool e responde ao usuário
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content