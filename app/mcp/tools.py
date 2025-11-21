TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_tts_audio",
            "description": "It generates audio from text. Use this tool ONLY when the user explicitly asks to 'hear', 'listen', 'speak' or 'generate audio'. Do NOT use this function if the user just asks to write or generate text sentences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "This is the text that will be transformed into audio.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pronunciation_practice",
            "description": "It allows the user to practice the pronunciation of a phrase or pronunciation in English.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Word or phrase that will be used to practice speaking.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_new_card",
            "description": "Creates a new flashcard in a specific deck.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "object",
                        "description": "Structured card content. Must contain a 'phrase | word' key.",
                        "properties": {
                            "phrase": {
                                "type": "string",
                                "description": "The phrase or word to be learned."
                            }
                        },
                        "required": ["phrase"]
                    },
                    "deck_id": {
                        "type": "string",
                        "description": "The public ID of the deck where the card will be created.",
                    },
                },
                "required": ["content", "deck_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_decks",
            "description": "Retrieves all available decks. Returns a list of decks with: public_id: str, name: str, type: Literal['speech', 'shadowing'], due_cards: int, total_cards: int",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_new_deck",
            "description": "Creates a new deck.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the new deck.",
                    },
                    "type": {
                        "type": "string",
                        "description": "Type of the deck: 'speech' or 'shadowing'. Default is 'speech'.",
                        "enum": ["speech", "shadowing"]
                    }
                },
                "required": ["name"],
            },
        },
    }
]