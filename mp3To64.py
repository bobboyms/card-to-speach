import base64

# Nome do arquivo de áudio MP3
nome_arquivo_audio = "audio.mp3"

# Abre o arquivo de áudio em modo de leitura binária ('rb')
with open(nome_arquivo_audio, "rb") as arquivo_mp3:
    # Lê o conteúdo binário do arquivo
    conteudo_binario = arquivo_mp3.read()

    # Codifica o conteúdo binário para Base64
    audio_base64_bytes = base64.b64encode(conteudo_binario)

    # Converte os bytes codificados em uma string de texto (opcional, mas comum)
    audio_base64_string = audio_base64_bytes.decode('utf-8')

    # Imprime a string Base64 resultante
    print(audio_base64_string)

    # Você também pode salvar a string Base64 em um arquivo de texto
    with open("audio_base64.txt", "w") as arquivo_texto:
        arquivo_texto.write(audio_base64_string)

print(f"\nO arquivo '{nome_arquivo_audio}' foi convertido para Base64 e salvo em 'audio_base64.txt'.")