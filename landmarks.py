"""
landmarks.py  -  MODULO COMPARTILHADO
Tudo que transforma "imagem da camera" em "vetor de numeros" mora aqui.
A coleta de dados e o reconhecimento em tempo real DEVEM importar daqui,
para que o modelo veja exatamente o mesmo tipo de dado no treino e no uso real.
"""
import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- MODELOS DO MEDIAPIPE (baixados automaticamente na 1a vez) ---
PASTA_MODELOS = "modelos"
URLS_MODELOS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

# --- FORMATO DO VETOR DE CADA FRAME ---
# Pontos do corpo (indices do MediaPipe Pose): 0 = nariz, 13 = cotovelo esq., 14 = cotovelo dir.
# Os ombros (11 e 12) NAO entram no vetor: eles viram a "regua" da normalizacao.
PONTOS_POSE = [0, 13, 14]
N_VALORES_MAO = 21 * 3                                   # 63 por mao (x, y, z)
N_FEATURES = 2 * N_VALORES_MAO + len(PONTOS_POSE) * 2    # 126 (maos) + 6 (corpo) = 132

CONEXOES_MAO = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]


def _ler_modelo(nome):
    """Garante que o .task existe (baixa se precisar) e devolve os bytes dele.
    Ler em bytes evita problemas com acentos/'c cedilha' no caminho do Windows."""
    candidatos = [os.path.join(PASTA_MODELOS, nome), nome]  # aceita o arquivo antigo na raiz
    for caminho in candidatos:
        if os.path.exists(caminho):
            with open(caminho, "rb") as f:
                return f.read()
    os.makedirs(PASTA_MODELOS, exist_ok=True)
    destino = os.path.join(PASTA_MODELOS, nome)
    print(f"Baixando {nome} (so na primeira vez)...")
    urllib.request.urlretrieve(URLS_MODELOS[nome], destino)
    with open(destino, "rb") as f:
        return f.read()


class Detector:
    """Roda MAOS + CORPO no mesmo frame, em modo VIDEO (rastreia entre frames)."""

    def __init__(self):
        opcoes_maos = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=_ler_modelo("hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        opcoes_pose = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=_ler_modelo("pose_landmarker_lite.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.maos = vision.HandLandmarker.create_from_options(opcoes_maos)
        self.pose = vision.PoseLandmarker.create_from_options(opcoes_pose)
        self._ultimo_ts = -1

    def processar(self, frame_bgr):
        """Recebe um frame do OpenCV (BGR). Devolve (resultado_maos, resultado_pose)."""
        # O modo VIDEO exige timestamps (ms) sempre crescentes
        ts = int(time.monotonic() * 1000)
        if ts <= self._ultimo_ts:
            ts = self._ultimo_ts + 1
        self._ultimo_ts = ts

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return (self.maos.detect_for_video(mp_image, ts),
                self.pose.detect_for_video(mp_image, ts))


def extrair_coordenadas(res_maos, res_pose, largura, altura):
    """
    Transforma o resultado do MediaPipe em um vetor de N_FEATURES numeros.

    NORMALIZACAO (o ponto-chave):
      - origem  = ponto no meio dos dois ombros
      - escala  = distancia entre os ombros
    Assim, os numeros dizem "onde a mao esta em relacao ao CORPO", e nao
    "onde a mao esta na imagem". Perto/longe da camera ou mais para o lado
    deixam de mudar o dado.

    Se o corpo (ombros) nao for detectado, devolve tudo zero (frame invalido).
    Se uma mao nao aparecer, os 63 valores dela ficam zero.
    """
    vazio = np.zeros(N_FEATURES)
    if not res_pose.pose_landmarks:
        return vazio

    pose = res_pose.pose_landmarks[0]
    ombro_esq, ombro_dir = pose[11], pose[12]

    # O MediaPipe entrega x,y entre 0 e 1 (proporcional a imagem). Como a imagem nao
    # e quadrada, convertemos para pixels antes de medir distancias, senao a escala fica torta.
    origem_x = (ombro_esq.x + ombro_dir.x) / 2 * largura
    origem_y = (ombro_esq.y + ombro_dir.y) / 2 * altura
    escala = np.hypot((ombro_esq.x - ombro_dir.x) * largura,
                      (ombro_esq.y - ombro_dir.y) * altura)
    if escala < 1e-6:
        return vazio

    def norm_xy(p):
        return [(p.x * largura - origem_x) / escala,
                (p.y * altura - origem_y) / escala]

    mao_esq = np.zeros(N_VALORES_MAO)
    mao_dir = np.zeros(N_VALORES_MAO)
    if res_maos.hand_landmarks and res_maos.handedness:
        for idx, mao in enumerate(res_maos.hand_landmarks):
            lado = res_maos.handedness[idx][0].category_name
            coords = []
            for p in mao:
                x, y = norm_xy(p)
                coords += [x, y, p.z * largura / escala]  # z ja e relativo ao pulso; so muda a escala
            if lado == 'Left':
                mao_esq = np.array(coords)
            else:
                mao_dir = np.array(coords)

    corpo = np.array([norm_xy(pose[i]) for i in PONTOS_POSE]).flatten()
    return np.concatenate([mao_esq, mao_dir, corpo])


def desenhar_maos(frame, res_maos):
    if res_maos.hand_landmarks:
        altura, largura, _ = frame.shape
        for mao in res_maos.hand_landmarks:
            pontos = [(int(p.x * largura), int(p.y * altura)) for p in mao]
            for px, py in pontos:
                cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
            for ini, fim in CONEXOES_MAO:
                cv2.line(frame, pontos[ini], pontos[fim], (0, 255, 0), 2)


def desenhar_corpo(frame, res_pose):
    """Desenha ombros (linha), nariz e cotovelos em azul, so para voce conferir se o corpo foi detectado."""
    if res_pose.pose_landmarks:
        altura, largura, _ = frame.shape
        pose = res_pose.pose_landmarks[0]
        px = lambda i: (int(pose[i].x * largura), int(pose[i].y * altura))
        cv2.line(frame, px(11), px(12), (255, 128, 0), 2)
        for i in [0, 11, 12, 13, 14]:
            cv2.circle(frame, px(i), 6, (255, 128, 0), -1)
