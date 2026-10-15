import cv2
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Descarrega automaticamente o modelo oficial de mãos da Google (apenas na 1.ª vez)
modelo_path = "hand_landmarker.task"
if not os.path.exists(modelo_path):
    print("A descarregar o modelo de mãos do MediaPipe...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, modelo_path)
    print("Modelo descarregado com sucesso!")

# 2. Lê o modelo em memória (evita erros com o 'çç' no caminho do Windows)
with open(modelo_path, "rb") as f:
    modelo_bytes = f.read()

# 3. Configura o detetor na nova API do MediaPipe
base_options = python.BaseOptions(model_asset_buffer=modelo_bytes)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detetor = vision.HandLandmarker.create_from_options(options)

# Ligações entre os 21 pontos da mão para desenhar o esqueleto
CONEXOES_MAO = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),        # Indicador
    (5, 9), (9, 10), (10, 11), (11, 12),   # Médio
    (9, 13), (13, 14), (14, 15), (15, 16), # Anelar
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Mindinho e palma
]

# 4. Tenta abrir a câmara 0; se não conseguir, tenta a câmara 1
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        print("Câmara ignorada ou não encontrada. Verifica se o Iriun mostra imagem.")
        break

    # Converte de BGR (OpenCV) para RGB (MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Deteta as mãos no frame atual
    resultado = detetor.detect(mp_image)

    # Se encontrar mãos, desenha os 21 pontos e as linhas
    if resultado.hand_landmarks:
        altura, largura, _ = frame.shape
        for mao in resultado.hand_landmarks:
            pontos_pixel = []
            for ponto in mao:
                px, py = int(ponto.x * largura), int(ponto.y * altura)
                pontos_pixel.append((px, py))
                cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

            for inicio, fim in CONEXOES_MAO:
                cv2.line(frame, pontos_pixel[inicio], pontos_pixel[fim], (0, 255, 0), 2)

    # Mostra a janela com o vídeo
    cv2.imshow('Teste MediaPipe - Prime Q para sair', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()