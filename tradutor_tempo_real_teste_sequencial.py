import os
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from landmarks import extrair_landmarks
from collections import deque

# caminho dos arquivos de treino
MODELO_PATH = os.path.join("modelos", "modelo_libras.keras")
ACOES_PATH = os.path.join("modelos", "acoes.json")

if not os.path.exists(MODELO_PATH) or not os.path.exists(ACOES_PATH):
    print("Erro: modelo treinado ou arquivo de ações não encontrado na pasta 'modelos/'.")
    raise SystemExit

print("Carregando o modelo de IA...")
model = load_model(MODELO_PATH)

with open(ACOES_PATH, "r", encoding="utf-8") as arquivo:
    ACOES = json.load(arquivo)

if not isinstance(ACOES, list) or not ACOES:
    print("Erro: o arquivo acoes.json não contém uma lista válida de ações.")
    raise SystemExit

if model.output_shape[-1] != len(ACOES):
    print("Erro: a quantidade de saídas do modelo não corresponde à quantidade de ações.")
    raise SystemExit

# configurações da analise
TAMANHO_SEQUENCIA = 30
INTERVALO_PREVISAO = 3
LIMITE_HISTORICO = 6

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: não foi possível abrir a webcam.")
    raise SystemExit

sequencia = deque(maxlen=TAMANHO_SEQUENCIA)
historico_previsoes = deque(maxlen=LIMITE_HISTORICO)

estado = "aguardando"
indice_alvo = 0
contador_frames = 0
resultados = {}


def reiniciar_analise():
    """Limpa os dados anteriores para analisar o próximo sinal do zero."""
    sequencia.clear()
    historico_previsoes.clear()


def desenhar_texto(imagem, mensagem, posicao, cor, tamanho=0.7):
    cv2.putText(
        imagem,
        mensagem,
        posicao,
        cv2.FONT_HERSHEY_SIMPLEX,
        tamanho,
        cor,
        2,
        cv2.LINE_AA
    )


print("\n--- TESTE DE RECONHECIMENTO DE LIBRAS ---")
print("I = iniciar | E = pausar/retomar | ESPAÇO = próximo sinal | Q = sair")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Erro ao acessar a webcam.")
        break

    landmarks, imagem_anotada = extrair_landmarks(frame)

    # Só coleta frames e executa previsões durante o teste ativo.
    if estado == "ativo":
        sequencia.append(landmarks)
        contador_frames += 1

        if len(sequencia) == TAMANHO_SEQUENCIA and contador_frames % INTERVALO_PREVISAO == 0:
            dados_entrada = np.expand_dims(np.array(sequencia), axis=0)
            previsoes = model(dados_entrada, training=False).numpy()[0]
            historico_previsoes.append(previsoes)

    if estado == "aguardando":
        desenhar_texto(imagem_anotada, "TESTE PRONTO",
                       (30, 45), (0, 255, 255), 0.9)
        desenhar_texto(imagem_anotada, "I = iniciar | Q = sair",
                       (30, 85), (255, 255, 255))

    elif estado == "ativo":
        sinal_alvo = ACOES[indice_alvo].upper()
        desenhar_texto(
            imagem_anotada,
            f"SINAL ESPERADO: {sinal_alvo}",
            (30, 45),
            (0, 255, 255),
            0.9
        )

        if len(sequencia) < TAMANHO_SEQUENCIA or not historico_previsoes:
            desenhar_texto(
                imagem_anotada,
                "Analisando movimento...",
                (30, 85),
                (0, 165, 255)
            )
        else:
            media_previsoes = np.mean(np.array(historico_previsoes), axis=0)
            confianca_alvo = float(media_previsoes[indice_alvo])
            indice_maior = int(np.argmax(media_previsoes))
            maior_sinal = ACOES[indice_maior].upper()
            maior_confianca = float(media_previsoes[indice_maior])

            desenhar_texto(
                imagem_anotada,
                f"{sinal_alvo}: {confianca_alvo * 100:.1f}%",
                (30, 85),
                (0, 255, 0),
                0.85
            )
            desenhar_texto(
                imagem_anotada,
                f"Maior identificacao: {maior_sinal} ({maior_confianca * 100:.1f}%)",
                (30, 125),
                (255, 255, 255),
                0.6
            )

        desenhar_texto(
            imagem_anotada,
            "ESPACO = proximo | E = pausar | Q = sair",
            (30, imagem_anotada.shape[0] - 25),
            (255, 255, 255),
            0.55
        )

    elif estado == "pausado":
        desenhar_texto(imagem_anotada, "TESTE PAUSADO",
                       (30, 45), (0, 165, 255), 0.9)
        desenhar_texto(imagem_anotada, "E = retomar | Q = sair",
                       (30, 85), (255, 255, 255))

    elif estado == "concluido":
        desenhar_texto(imagem_anotada, "TESTE CONCLUIDO!",
                       (30, 45), (0, 255, 0), 0.9)
        desenhar_texto(
            imagem_anotada, "Confira o resumo no terminal.", (30, 85), (255, 255, 255))

    cv2.imshow("Teste de Libras - Tempo Real", imagem_anotada)

    tecla = cv2.waitKey(10) & 0xFF

    if tecla in (ord("q"), ord("Q")):
        break
    elif tecla in (ord("i"), ord("I")) and estado == "aguardando":
        estado = "ativo"
        contador_frames = 0
        reiniciar_analise()

    elif tecla in (ord("e"), ord("E")) and estado == "ativo":
        estado = "pausado"
        reiniciar_analise()

    elif tecla in (ord("e"), ord("E")) and estado == "pausado":
        estado = "ativo"
        contador_frames = 0
        reiniciar_analise()

    elif tecla == ord(" ") and estado == "ativo":
        # Guarda a pontuação média do sinal atual antes de avançar.
        if historico_previsoes:
            media_previsoes = np.mean(np.array(historico_previsoes), axis=0)
            resultados[ACOES[indice_alvo]] = float(
                media_previsoes[indice_alvo])
        else:
            resultados[ACOES[indice_alvo]] = None

        indice_alvo += 1

        if indice_alvo >= len(ACOES):
            estado = "concluido"
            print("\n--- RESULTADO DO TESTE ---")
            for acao in ACOES:
                pontuacao = resultados.get(acao)
                if pontuacao is None:
                    print(f"{acao.upper()}: sem leitura suficiente")
                else:
                    print(f"{acao.upper()}: {pontuacao * 100:.1f}%")
cap.release()
cv2.destroyAllWindows()
