# main.py
import os
import json
import re
from openai import OpenAI

# Certifique-se de ter essas importações corretas no seu projeto
from app.mcp.mcp_client import call_generate_tts_audio_sync, call_pronunciation_practice_sync, call_create_new_card_sync, call_get_all_decks_sync, call_create_new_deck_sync
# Se tiver uma função para pronúncia, importe-a aqui também, ex:
# from app.mcp.mcp_client import call_pronunciation_practice_sync
from app.mcp.tools import TOOLS
from app import config


class ChatService:
    def __init__(self, model_name=None):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in environment variables.")

        self.client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name or config.OPENAI_MODEL

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

    def generate_answer_stream(self, history, user_message: str, user_id: str):
        """
        history: lista de dicts {"role": "user"|"assistant", "content": str}
        user_message: mensagem atual do usuário
        user_id: ID do usuário autenticado

        Gera pedaços de texto (chunks) da resposta final para streaming.
        """
        
        # Tenta extrair o deck_id do histórico
        active_deck_id = self._extract_deck_id(history)

        system_prompt = (
            "Você é um assistente útil focado em aprendizado de idiomas e SEMPRE responde em português brasileiro, "
            "a menos que o usuário peça explicitamente a resposta em outro idioma. "
            "Você pode usar frases de exemplo em inglês quando isso for útil.\n\n"

            "REGRAS GERAIS SOBRE FERRAMENTAS:\n"
            "- Só chame funções (ferramentas) quando isso for realmente necessário para ajudar o usuário.\n"
            "- Se for possível responder apenas com texto, responda apenas com texto.\n\n"

            "FUNÇÃO 'generate_tts_audio':\n"
            "- NÃO use a função 'generate_tts_audio' a menos que o usuário peça explicitamente para ouvir algo, "
            "gerar áudio ou peça algo como \"fale\", \"pronuncie\" ou \"quero ouvir\".\n"
            "- Se o usuário pedir apenas exemplos de frases, apenas escreva o texto, sem chamar a função.\n"
            "- Quando você usar a função 'generate_tts_audio', ela retornará um JSON com o campo 'file_path'.\n"
            "- Sempre que usar essa função, termine sua resposta com uma linha EXATAMENTE neste formato:\n"
            "AUDIO_FILE: <file_path_do_áudio>\n"
            "- Não coloque NENHUM texto depois dessa linha. Toda explicação deve vir antes.\n\n"

            "FUNÇÃO 'pronunciation_practice':\n"
            "- Use a função 'pronunciation_practice' quando o usuário quiser praticar a pronúncia "
            "de uma palavra ou frase em inglês.\n"
            "- A função retornará um JSON com o campo 'practice'.\n"
            "- Sempre que usar essa função, termine sua resposta com uma linha EXATAMENTE neste formato:\n"
            "START_PRACTICE: <text>\n"
            "- Substitua <text> exatamente pelo valor de 'practice' retornado no JSON.\n"
            "- Não coloque NENHUM texto depois dessa linha. Toda explicação deve vir antes.\n\n"

            "FUNÇÃO 'create_new_card':\n"
            "- Para chamar a função 'create_new_card', você DEVE fornecer um 'deck_id'.\n"
            "- Se o usuário não tiver especificado em qual deck criar o card, PERGUNTE antes "
            "em qual deck ele deseja criar o card, e só então chame a função.\n"
            "- Nunca invente ou suponha um 'deck_id'.\n"
            "- A função retornará um JSON. Se o JSON contiver 'error', informe claramente ao usuário qual foi o erro.\n"
            "- Se a função retornar os dados do card criado, confirme a criação com sucesso e, se fizer sentido, "
            "resuma os dados principais do card para o usuário.\n\n"

            "FUNÇÃO 'get_all_decks':\n"
            "- Use a função 'get_all_decks' sempre que o usuário quiser saber quais decks existem "
            "ou quando você precisar ajudar o usuário a escolher um deck.\n"
            "- O retorno da função é uma lista de objetos com os campos: "
            "public_id: str, name: str, type: Literal['speech', 'shadowing'], due_cards: int, total_cards: int.\n"
            "- Explique em detalhes os decks disponíveis em português claro, incluindo:\n"
            "  • o ID do deck (public_id),\n"
            "  • o nome do deck (name),\n"
            "  • o tipo (type) e para que ele é mais indicado,\n"
            "  • quantos cards estão para revisão (due_cards) e o total de cards (total_cards).\n\n"

            "FUNÇÃO 'create_new_deck':\n"
            "- Use a função 'create_new_deck' quando o usuário quiser criar um novo deck.\n"
            "- Pergunte o nome do deck se não for fornecido.\n"
            "- O tipo (type) é opcional e padrão é 'speech'.\n"
            "- A função retorna os dados do deck criado.\n"
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
                card_result = call_create_new_card_sync(content, deck_id, user_id)
                tool_result_content = json.dumps(card_result)

            elif name == "get_all_decks":
                decks_result = call_get_all_decks_sync(user_id)
                tool_result_content = json.dumps(decks_result)

            elif name == "create_new_deck":
                deck_name = args.get("name")
                deck_type = args.get("type", "speech")
                deck_result = call_create_new_deck_sync(deck_name, user_id, deck_type)
                tool_result_content = json.dumps(deck_result)

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