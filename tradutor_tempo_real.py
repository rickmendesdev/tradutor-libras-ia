import os
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from landmarks import extrair_landmarks

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
limiar_confianca = 0.75  # Só exibe se a certeza for maior que 75%

print("\n--- TRADUTOR EM TEMPO REAL ATIVADO ---")
print("Pressione 'q' na janela do vídeo para fechar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a webcam.")
        break

    # Extrair os landmarks do frame atual usando o módulo de landmarks do projeto
    landmarks, imagem_anotada = extrair_landmarks(frame)

    # Guardar na sequência temporal
    sequencia.append(landmarks)
    # Manter apenas os últimos 30 frames (1 segundo de vídeo)
    sequencia = sequencia[-30:]

    # Quando completarmos 30 frames, a IA faz a predição
    if len(sequencia) == 30:
        dados_entrada = np.expand_dims(np.array(sequencia), axis=0)
        previsoes = model.predict(dados_entrada, verbose=0)[0]
        classe_predita = np.argmax(previsoes)
        confianca = previsoes[classe_predita]

        # Verificar se a confiança passa do limiar definido
        if confianca >= limiar_confianca:
            sinal_reconhecido = ACOES[classe_predita].upper()
            texto = f"{sinal_reconhecido} ({confianca * 100:.1f}%)"
            cor = (0, 255, 0)  # Verde se reconheceu com segurança
        else:
            texto = "Analisando..."
            cor = (0, 165, 255)  # Laranja se estiver abaixo da confiança

        # Desenhar o resultado na tela
        cv2.putText(imagem_anotada, texto, (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2, cv2.LINE_AA)

    # Exibir a janela com a câmera
    cv2.imshow('Tradutor de Libras - Tempo Real', imagem_anotada)

    # Pressionar 'q' para sair
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()