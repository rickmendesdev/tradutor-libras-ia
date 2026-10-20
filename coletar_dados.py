import glob
import os
import time

import cv2
import numpy as np

from landmarks import Detector, extrair_coordenadas, desenhar_maos, desenhar_corpo

# --- 1. CONFIGURACOES DO DATASET ---
# TROQUE no PC de cada um (ex.: o nome do seu amigo). Vai no nome dos arquivos.
PESSOA = "izanagi"
PASTA_DADOS = "dados_libras"
# TROQUE pelos sinais que voces validaram
ACOES = ['feliz', 'triste', 'raiva', 'medo', 'amor', 'parado']
NUM_SEQUENCIAS = 30         # quantas repeticoes de cada sinal ESTA pessoa vai gravar
TAMANHO_SEQUENCIA = 30      # frames por repeticao (~1 segundo)
SEGUNDOS_CONTAGEM = 2       # tempo de "prepare-se" antes de gravar
# se mais de 20% dos frames perderem o corpo, a repeticao e descartada
MAX_FRAMES_INVALIDOS = 0.2
JANELA = 'Coleta de Dados - Libras'


def caminho_sequencia(acao, n):
    return os.path.join(PASTA_DADOS, acao, f"{PESSOA}_{n}.npy")


def contar_feitas(acao):
    return len(glob.glob(os.path.join(PASTA_DADOS, acao, f"{PESSOA}_*.npy")))


def proximo_indice(acao):
    n = 0
    while os.path.exists(caminho_sequencia(acao, n)):
        n += 1
    return n


def texto(frame, msg, y, cor=(0, 255, 255), tamanho=0.7):
    # Atencao: as fontes do OpenCV nao desenham acentos. Use texto sem acento na tela.
    cv2.putText(frame, msg, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                tamanho, cor, 2, cv2.LINE_AA)


def sair(cap, motivo=""):
    cap.release()
    cv2.destroyAllWindows()
    if motivo:
        print(motivo)
    raise SystemExit


def ler_frame(cap, detector):
    """Le a camera, detecta maos+corpo e desenha o esqueleto no frame."""
    ok, frame = cap.read()
    if not ok:
        sair(cap, "Falha ao ler a camera.")
    res_maos, res_pose = detector.processar(frame)
    desenhar_maos(frame, res_maos)
    desenhar_corpo(frame, res_pose)
    return frame, res_maos, res_pose


# --- 2. CAMERA E DETECTOR ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
detector = Detector()

for acao in ACOES:
    os.makedirs(os.path.join(PASTA_DADOS, acao), exist_ok=True)
    feitas = contar_feitas(acao)
    if feitas >= NUM_SEQUENCIAS:
        print(f"'{acao}' ja tem {feitas} sequencias de {PESSOA}. Pulando.")
        continue

    # Tela de espera: voce descansa e comeca quando quiser
    while True:
        frame, _, _ = ler_frame(cap, detector)
        texto(
            frame, f"Proximo sinal: {acao.upper()}  ({feitas}/{NUM_SEQUENCIAS})", 30)
        texto(frame, "ESPACO = comecar | Q = sair", 60, (255, 255, 255))
        cv2.imshow(JANELA, frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            sair(cap)
        if tecla == 32:
            break

    while feitas < NUM_SEQUENCIAS:
        # --- Contagem: video AO VIVO, nada e gravado ---
        inicio = time.monotonic()
        while time.monotonic() - inicio < SEGUNDOS_CONTAGEM:
            frame, _, _ = ler_frame(cap, detector)
            restante = SEGUNDOS_CONTAGEM - (time.monotonic() - inicio)
            texto(
                frame, f"PREPARE-SE: {acao.upper()}  ({restante:.0f}s)", 30, (0, 255, 0))
            cv2.imshow(JANELA, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sair(cap)

        # --- Gravacao: TODOS os 30 frames sao gravados ---
        sequencia = []
        for _ in range(TAMANHO_SEQUENCIA):
            frame, res_maos, res_pose = ler_frame(cap, detector)
            altura, largura, _ = frame.shape
            sequencia.append(extrair_coordenadas(
                res_maos, res_pose, largura, altura))
            texto(
                frame, f"GRAVANDO: {acao.upper()}  {feitas + 1}/{NUM_SEQUENCIAS}", 30, (0, 0, 255))
            cv2.imshow(JANELA, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sair(cap)

        sequencia = np.array(sequencia)               # formato: (30, 132)
        frames_invalidos = np.mean(~sequencia.any(axis=1))
        if frames_invalidos > MAX_FRAMES_INVALIDOS:
            print(
                f"Repeticao descartada ({frames_invalidos:.0%} dos frames sem corpo detectado). Refazendo.")
            continue

        np.save(caminho_sequencia(acao, proximo_indice(acao)),
                sequencia)  # 1 arquivo por repeticao
        feitas += 1

cap.release()
cv2.destroyAllWindows()
print("Coleta finalizada com sucesso!")
