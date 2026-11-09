import os
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from landmarks import extrair_landmarks
import time
from collections import deque, Counter

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

# NOVO: So aceita um sinal quando ha consenso entre as ultimas respostas do modelo


def estabilizar(ultimas, acoes, limiar=0.15, votos=5):
    """Devolve o nome do sinal quando ha consenso; senao, None."""
    if len(ultimas) < ultimas.maxlen:
        return None
    validas = [i if c >= limiar else -1 for i,
               c in ultimas]  # -1 = sem certeza
    indice, qtd = Counter(validas).most_common(1)[0]
    if indice == -1 or qtd < votos:
        return None
    return acoes[indice]


# Inicializar a webcam
cap = cv2.VideoCapture(0)

sequencia = []
limiar_confianca = 0.75

ultimas = deque(maxlen=6)
aceito = None
resultado, resultado_conf = "", 0.0  # o que esta sendo mostrado na tela
mostar_ate = 0.0
contador_frames = 0  # conta frames para prever a cada 3

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

    # alteração importante:
    contador_frames += 1

    if len(sequencia) == 30 and contador_frames % 3 == 0:
        dados_entrada = np.expand_dims(np.array(sequencia), axis=0)
        previsoes = model(dados_entrada, training=False).numpy()[
            0]  # ALTERADO: mais rapido que predict()
        classe_predita = int(np.argmax(previsoes))
        confianca = float(previsoes[classe_predita])
        ultimas.append((classe_predita, confianca))

        # Verificar se a confiança passa do limiar definido
        sinal = estabilizar(ultimas, ACOES, limiar_confianca)
        if sinal == 'parado':
            aceito = None
        elif sinal is not None and sinal != aceito:
            resultado, resultado_conf = sinal.upper(), confianca
            aceito = sinal
            mostar_ate = time.monotonic() + 1.5

        if time.monotonic() < mostar_ate:
            texto = f"{resultado} ({resultado_conf * 100:.0f}%)"
            cor = (0, 255, 0)  # verde: sinal reconhecido
        else:
            texto = "Aguardando sinal..."
            cor = (0, 165, 255)  # laranja
        cv2.putText(imagem_anotada, texto, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2, cv2.LINE_AA)

    # Exibir a janela com a câmera
    cv2.imshow('Tradutor de Libras - Tempo Real', imagem_anotada)

    # Pressionar 'q' para sair
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
