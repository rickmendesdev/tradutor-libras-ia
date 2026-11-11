
import os
import json
import cv2
import numpy as np
from collections import deque, Counter
from tensorflow.keras.models import load_model
from landmarks import extrair_landmarks
from database import registrar_reconhecimento

# Caminhos dos arquivos gerados no treino
MODELO_PATH = os.path.join("modelos", "modelo_libras.keras")
ACOES_PATH = os.path.join("modelos", "acoes.json")

if not os.path.exists(MODELO_PATH) or not os.path.exists(ACOES_PATH):
    print("Erro: O modelo treinado ou o arquivo de ações não foram encontrados na pasta 'modelos/'!")
    exit()

# Carregar o modelo e o dicionário de ações
print("Carregando o modelo de IA...")
model = load_model(MODELO_PATH)

with open(ACOES_PATH, "r", encoding="utf-8") as f:
    ACOES = json.load(f)

print(f"Sinais reconhecidos pelo modelo: {ACOES}")

# Inicializar a webcam
cap = cv2.VideoCapture(0)

sequencia = []
limiar_confianca = 0.75

# Histórico curto para suavizar as previsões
historico_previsoes = deque(maxlen=5)

# Guarda o último sinal registrado para evitar inserções repetidas
ultimo_sinal_registrado = None

print("\n--- TRADUTOR EM TEMPO REAL ATIVADO (MODO FLUIDO) ---")
print("Pressione 'q' na janela do vídeo para fechar.")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Erro ao acessar a webcam.")
        break

    # Extrair os landmarks do frame atual
    landmarks, imagem_anotada = extrair_landmarks(frame)

    # Guardar os últimos 30 frames
    sequencia.append(landmarks)
    sequencia = sequencia[-30:]

    if len(sequencia) == 30:
        dados_entrada = np.expand_dims(
            np.array(sequencia), axis=0
        )

        # Realizar a predição
        previsoes = model(dados_entrada, training=False).numpy()[0]
        classe_predita = int(np.argmax(previsoes))
        confianca = float(previsoes[classe_predita])

        # Guardar previsões que passam do limiar de confiança
        if confianca >= limiar_confianca:
            historico_previsoes.append(classe_predita)
        else:
            historico_previsoes.append(-1)

        # Considerar apenas previsões válidas
        validas = [
            c for c in historico_previsoes if c != -1
        ]

        texto = "Aguardando sinal..."
        cor = (0, 165, 255)

        if validas:
            mais_comum, qtd = Counter(validas).most_common(1)[0]

            # Exigir pelo menos 3 previsões iguais
            if qtd >= 3:
                sinal = ACOES[mais_comum]
                confianca_exibida = float(previsoes[mais_comum])

                # Sinal neutro: libera o próximo reconhecimento
                if sinal == "parado":
                    ultimo_sinal_registrado = None
                    texto = "Aguardando sinal..."

                else:
                    # Registrar somente quando houver um novo sinal
                    if sinal != ultimo_sinal_registrado:
                        texto_voz = registrar_reconhecimento(sinal)
                        ultimo_sinal_registrado = sinal

                    # Utilizar o texto retornado pelo banco
                    if sinal == ultimo_sinal_registrado:
                        if texto_voz is not None:
                            sinal_exibido = texto_voz.upper()
                        else:
                            sinal_exibido = sinal.upper()

                    texto = (
                        f"{sinal_exibido} "
                        f"({confianca_exibida * 100:.1f}%)"
                    )
                    cor = (0, 255, 0)

        # Desenhar o resultado na tela
        cv2.putText(
            imagem_anotada,
            texto,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            cor,
            2,
            cv2.LINE_AA
        )

    # Exibir a janela com a câmera
    cv2.imshow(
        'Tradutor de Libras - Tempo Real',
        imagem_anotada
    )

    # Pressionar 'q' para sair
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
