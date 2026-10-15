import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. CONFIGURAÇÕES DO DATASET ---
PASTA_DADOS = "dados_libras"
ACOES = np.array(['oi', 'tudo_bem', 'parado']) # Sinais que vamos ensinar à IA
NUM_SEQUENCIAS = 30        # Quantas vezes você vai repetir cada sinal (30 vídeos)
TAMANHO_SEQUENCIA = 30     # Quantos frames tem cada movimento

# Cria as pastas para salvar os arquivos (Ex: dados_libras/oi/0, dados_libras/oi/1...)
for acao in ACOES:
    for seq in range(NUM_SEQUENCIAS):
        os.makedirs(os.path.join(PASTA_DADOS, acao, str(seq)), exist_ok=True)

# --- 2. INICIALIZA O MEDIAPIPE ---
with open("hand_landmarker.task", "rb") as f:
    modelo_bytes = f.read()

base_options = python.BaseOptions(model_asset_buffer=modelo_bytes)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detetor = vision.HandLandmarker.create_from_options(options)

CONEXOES_MAO = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def extrair_coordenadas(resultado):
    """Extrai 126 valores (X, Y, Z de 21 pontos para 2 mãos). Preenche com 0 se a mão não aparecer."""
    mao_esq = np.zeros(21 * 3)
    mao_dir = np.zeros(21 * 3)

    if resultado.hand_landmarks and resultado.handedness:
        for idx, mao in enumerate(resultado.hand_landmarks):
            lado = resultado.handedness[idx][0].category_name
            coords = np.array([[p.x, p.y, p.z] for p in mao]).flatten()
            if lado == 'Left':
                mao_esq = coords
            else:
                mao_dir = coords

    return np.concatenate([mao_esq, mao_dir])

def desenhar_maos(frame, resultado):
    if resultado.hand_landmarks:
        altura, largura, _ = frame.shape
        for mao in resultado.hand_landmarks:
            pontos_pixel = [(int(p.x * largura), int(p.y * altura)) for p in mao]
            for px, py in pontos_pixel:
                cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
            for inicio, fim in CONEXOES_MAO:
                cv2.line(frame, pontos_pixel[inicio], pontos_pixel[fim], (0, 255, 0), 2)

# --- 3. CAPTURA DOS MOVIMENTOS ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

print("Iniciando coleta! Prepare-se em frente à câmera...")

for acao in ACOES:
    for seq in range(NUM_SEQUENCIAS):
        for frame_num in range(TAMANHO_SEQUENCIA):
            sucesso, frame = cap.read()
            if not sucesso:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            resultado = detetor.detect(mp_image)

            desenhar_maos(frame, resultado)

            # No frame 0 de cada vídeo, dá uma pausa de 2 segundos para você posicionar a mão
            if frame_num == 0:
                cv2.putText(frame, f'PREPARE-SE: {acao.upper()}', (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, f'Sinal: {acao} | Video {seq + 1}/{NUM_SEQUENCIAS}', (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Coleta de Dados - Libras', frame)
                cv2.waitKey(2000) # Pausa de 2 segundos entre cada repetição
            else:
                cv2.putText(frame, f'GRAVANDO: {acao} | Video {seq + 1}/{NUM_SEQUENCIAS}', (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Coleta de Dados - Libras', frame)

            # Extrai os 126 pontos e salva no arquivo .npy correspondente
            pontos = extrair_coordenadas(resultado)
            caminho_arquivo = os.path.join(PASTA_DADOS, acao, str(seq), str(frame_num))
            np.save(caminho_arquivo, pontos)

            # Permite cancelar apertando 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

cap.release()
cv2.destroyAllWindows()
print("Coleta finalizada com sucesso!")