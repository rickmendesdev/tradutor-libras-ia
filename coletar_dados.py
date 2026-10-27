import glob
import os
import sys
import time

import cv2
import numpy as np

# Reutiliza o detector, a extracao de coordenadas e os desenhos do modulo compartilhado.
from landmarks import (Detector, extrair_coordenadas, desenhar_maos, desenhar_corpo,
                       N_VALORES_MAO)


def obter_pessoa():
    nome = sys.argv[1] if len(sys.argv) > 1 else input(
        "Nome de quem esta gravando (ex.: Vinicius): "
    )
    nome = nome.strip().lower()

    if not (nome.isascii() and nome.isalnum()):
        raise SystemExit(
            "Use so letras e numeros no nome "
            "(sem acento, espaco ou '_')."
        )

    return nome

# --- 1. CONFIGURACOES DO DATASET ---
# TROQUE no PC de cada um (ex.: o nome do seu amigo). Vai no nome dos arquivos.
# PESSOA = "Vinicius"
# PASTA_DADOS = "dados_libras"


PESSOA = obter_pessoa()
PASTA_DADOS = "dados_libras"
# OBS:DEVEMOS CATALOGAR ESSES SINAIS LOGO
ACOES = ['rindo', 'triste', 'vergonha', 'nojo', 'inveja',
         'medo', 'feliz', 'confusao', 'nervoso', 'raiva',
         'bravo', 'inimigo', 'odio', 'idiota', 'fingir',
         'deboche', 'parado']
# Com 10 repeticoes reais, o treinar_modelo.py multiplica para 40 automaticamente na memoria!
NUM_SEQUENCIAS = 10       # quantas repeticoes de cada sinal ESTA pessoa vai gravar
TAMANHO_SEQUENCIA = 30      # frames por repeticao (~1 segundo)
SEGUNDOS_CONTAGEM = 2       # tempo de "prepare-se" antes de gravar
# se mais de 20% dos frames perderem o corpo, a repeticao e descartada
MAX_FRAMES_INVALIDOS = 0.2
# Nos sinais com mão, pelo menos 70% dos frames precisam ter alguma mão detectada
MIN_FRAMES_COM_MAO = 0.7
ACOES_SEM_MAO = ['parado']
JANELA = 'Coleta de Dados - Libras'


def caminho_sequencia(acao, n):
    """Monta o caminho onde sera salvo o exemplo numero n da acao."""
    return os.path.join(PASTA_DADOS, acao, f"{PESSOA}_{n}.npy")


def contar_feitas(acao):
    """Conta as sequencias reais ja gravadas para esta pessoa e acao."""
    arquivos = glob.glob(
        os.path.join(PASTA_DADOS, acao, f"{PESSOA}_*.npy")
    )

    return len([
        f for f in arquivos
        if "_aug_" not in os.path.basename(f)
    ])


def proximo_indice(acao):
    """Busca o primeiro indice de arquivo que ainda nao existe para evitar sobrescrita."""
    n = 0
    while os.path.exists(
            caminho_sequencia(acao, n)):
        n += 1
    return n


def texto(frame, msg, y, cor=(0, 255, 255), tamanho=0.7):
    """Desenha uma instrucao na imagem da camera."""
    # Atencao: as fontes do OpenCV nao desenham acentos. Use texto sem acento na tela.
    cv2.putText(frame, msg, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                tamanho, cor, 2, cv2.LINE_AA)


def indicador_mao(frame, res_maos):
    """Mostra se alguma mão esta sendo detectada"""
    tem_mao = bool(res_maos.hand_landmarks)
    texto(
        frame,
        "MAO OK" if tem_mao else "SEM MAO!",
        60,
        (0, 255, 0) if tem_mao else (0, 0, 255)
    )


def sair(cap, motivo=""):
    """Libera os recursos da camera e janelas antes de encerrar a coleta."""
    cap.release()
    cv2.destroyAllWindows()
    if motivo:
        print(motivo)
    raise SystemExit


def ler_frame(cap, detector):
    """Le a camera, detecta maos+corpo e desenha o esqueleto no frame."""
    # Captura um frame e encerra com uma mensagem se a camera falhar.
    ok, frame = cap.read()
    if not ok:
        sair(cap, "Falha ao ler a camera.")
    # Detecta maos e corpo e desenha os pontos para dar retorno visual a quem grava.
    res_maos, res_pose = detector.processar(frame)
    desenhar_maos(frame, res_maos)
    desenhar_corpo(frame, res_pose)
    return frame, res_maos, res_pose


# --- 2. CAMERA E DETECTOR ---
# Tenta a camera padrao e usa a segunda entrada como alternativa.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
detector = Detector()

# Para cada sinal, prepara a pasta e pula a coleta se a meta ja foi atingida.
for acao in ACOES:
    os.makedirs(os.path.join(PASTA_DADOS, acao), exist_ok=True)
    feitas = contar_feitas(acao)
    if feitas >= NUM_SEQUENCIAS:
        print(f"'{acao}' ja tem {feitas} sequencias de {PESSOA}. Pulando.")
        continue

    # Tela de espera: voce descansa e comeca quando quiser
    # Aguarda a pessoa iniciar a coleta com ESPACO ou encerrar com Q.
    while True:
        frame, res_maos, _ = ler_frame(cap, detector)
        indicador_mao(frame, res_maos)
        # tentativa de correção para problema de execução

        texto(frame, f"Proximo sinal: {acao.upper()} ({feitas}/{NUM_SEQUENCIAS})", 30)
        texto(frame, "ESPACO = comeca | q = sair", 90)

        cv2.imshow(JANELA, frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord(' '):
            break
        elif tecla == ord('q'):
            sair(cap, "Coleta encerrada pelo usuario")

        # texto(
        #   frame,
        #   f"Proximo sinal: {acao.upper()} ({feitas}/{NUM_SEQUENCIAS})",
        #    30
        # )
        # texto(
        #    frame,
        #    "ESPACO = comecar | q = sair",
        #    90,
        #    (255, 255, 255)
        # )

    # Repete contagem e gravacao ate atingir o total configurado para este sinal.
    while feitas < NUM_SEQUENCIAS:
        # --- Contagem: video AO VIVO, nada e gravado ---
        inicio = time.monotonic()
        while time.monotonic() - inicio < SEGUNDOS_CONTAGEM:
            frame, res_maos, _ = ler_frame(cap, detector)
            indicador_mao(frame, res_maos)
            restante = SEGUNDOS_CONTAGEM - (time.monotonic() - inicio)
            texto(
                frame, f"PREPARE-SE: {acao.upper()}  ({restante:.0f}s)", 30, (0, 255, 0))
            cv2.imshow(JANELA, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sair(cap)

        # --- Gravacao: TODOS os 30 frames sao gravados ---
        # Guarda as coordenadas extraidas de cada frame, na ordem em que foram capturadas.
        sequencia = []
        for _ in range(TAMANHO_SEQUENCIA):
            frame, res_maos, res_pose = ler_frame(cap, detector)
            indicador_mao(frame, res_maos)
            altura, largura, _ = frame.shape
            sequencia.append(extrair_coordenadas(
                res_maos, res_pose, largura, altura))
            texto(
                frame, f"GRAVANDO: {acao.upper()}  {feitas + 1}/{NUM_SEQUENCIAS}", 30, (0, 0, 255))
            cv2.imshow(JANELA, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sair(cap)

        # Converte a lista em matriz; cada linha e um frame e cada coluna uma coordenada.
        sequencia = np.array(sequencia)               # formato: (30, 132)
        
        # Descarta repeticoes em que muitas linhas estao zeradas por falta de deteccao do corpo.
        frames_invalidos = np.mean(~sequencia.any(axis=1))
        if frames_invalidos > MAX_FRAMES_INVALIDOS:
            print(
                f"Repeticao descartada ({frames_invalidos:.0%} dos frames sem corpo detectado). Refazendo.")
            continue

        # Descarta repeticoes de sinais com mao onde a mao quase nao apareceu no quadro.
        if acao not in ACOES_SEM_MAO:
            frames_com_mao = np.mean(sequencia[:, :2 * N_VALORES_MAO].any(axis=1))
            if frames_com_mao < MIN_FRAMES_COM_MAO:
                print(
                    f"Repeticao descartada (mao detectada em apenas {frames_com_mao:.0%} dos frames). Refazendo.")
                continue

        # Salva cada repeticao em seu proprio arquivo NumPy para uso posterior no treinamento.
        np.save(caminho_sequencia(acao, proximo_indice(acao)),
                sequencia)  # 1 arquivo por repeticao
        feitas += 1

cap.release()
cv2.destroyAllWindows()
print("Coleta finalizada com sucesso!")