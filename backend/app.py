import os
import json
import time
import asyncio
import cv2
import numpy as np
from collections import deque, Counter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

# Importa os módulos que já estão na pasta backend
from .landmarks import extrair_landmarks
from .database import registrar_reconhecimento

app = FastAPI(title="API Tradutor Libras", version="1.0")

# Configuração de CORS para permitir requisições do front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique a origem exata
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminhos dos modelos na raiz do projeto (subindo um nível a partir de backend/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_PATH = os.path.join(BASE_DIR, "modelos", "modelo_libras.keras")
ACOES_PATH = os.path.join(BASE_DIR, "modelos", "acoes.json")

# Carregar modelo e ações ao iniciar o servidor
if not os.path.exists(MODELO_PATH) or not os.path.exists(ACOES_PATH):
    raise RuntimeError(
        "Modelo treinado ou arquivo acoes.json não encontrados na pasta 'modelos/'!")

print("Carregando modelo de IA no servidor...")
model = load_model(MODELO_PATH)

with open(ACOES_PATH, "r", encoding="utf-8") as f:
    ACOES = json.load(f)

print(f"Ações carregadas: {ACOES}")


def _decodificar_e_extrair(data: bytes):
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    landmarks, _ = extrair_landmarks(frame, desenhar=False)
    return landmarks


def _inferir_modelo(dados_entrada: np.ndarray):
    return model(dados_entrada, training=False).numpy()[0]


@app.get("/")
def home():
    return {"status": "API do Tradutor de Libras está ativa e operacional!"}


# Endpoint WebSocket para processamento em tempo real fluido
@app.websocket("/ws/traduzir")
async def websocket_tradutor(websocket: WebSocket):
    await websocket.accept()

    # Estado da sessão para este cliente específico
    sequencia = []
    historico_previsoes = deque(maxlen=5)
    ultimo_sinal_registrado = None
    ultimo_texto_voz = ""
    limiar_confianca = 0.75

    try:
        while True:
            t_inicio_frame = time.perf_counter()

            # Recebe os bytes da imagem (frame) enviados pelo front-end via WebSocket
            data = await websocket.receive_bytes()

            t_decode_lm_0 = time.perf_counter()
            landmarks = await asyncio.to_thread(_decodificar_e_extrair, data)
            t_decode_lm_1 = time.perf_counter()

            if landmarks is None:
                continue

            sequencia.append(landmarks)
            # Mantém a janela deslizante de 30 frames
            sequencia = sequencia[-30:]

            resposta_json = {
                "sinal": "Aguardando sinal...",
                "confianca": 0.0,
                "texto_voz": "",
                "novo_sinal": False
            }

            tempo_modelo_ms = 0.0

            if len(sequencia) == 30:
                dados_entrada = np.expand_dims(np.array(sequencia), axis=0)

                # Inferência do modelo
                t_mod_0 = time.perf_counter()
                previsoes = await asyncio.to_thread(_inferir_modelo, dados_entrada)
                t_mod_1 = time.perf_counter()
                tempo_modelo_ms = (t_mod_1 - t_mod_0) * 1000

                classe_predita = int(np.argmax(previsoes))
                confianca = float(previsoes[classe_predita])

                if confianca >= limiar_confianca:
                    historico_previsoes.append(classe_predita)
                else:
                    historico_previsoes.append(-1)

                # Conta quantas previsões indicaram ausência de sinal confiável
                quantidade_neutras = sum(
                    classe == -1 for classe in historico_previsoes
                )

                # Três previsões inválidas liberam um novo gesto
                validas = []

                if quantidade_neutras >= 3:
                    ultimo_sinal_registrado = None
                    ultimo_texto_voz = ""
                    historico_previsoes.clear()
                else:
                    # Remove as previsões inválidas para realizar a votação
                    validas = [
                        classe for classe in historico_previsoes
                        if classe != -1
                    ]

                if validas:
                    mais_comum, qtd = Counter(validas).most_common(1)[0]

                    if qtd >= 3:
                        sinal = ACOES[mais_comum]
                        confianca_exibida = float(previsoes[mais_comum])

                        novo_sinal = sinal != ultimo_sinal_registrado

                        if novo_sinal:
                            texto_voz = await asyncio.to_thread(
                                registrar_reconhecimento, sinal
                            )
                            if texto_voz:
                                ultimo_texto_voz = texto_voz
                            else:
                                ultimo_texto_voz = f"Sinal identificado: {sinal}."

                            ultimo_sinal_registrado = sinal

                        resposta_json = {
                            "sinal": sinal.upper(),
                            "confianca": round(confianca_exibida * 100, 1),
                            "texto_voz": ultimo_texto_voz,
                            "novo_sinal": novo_sinal
                        }

            # Envia o resultado de volta para o JavaScript no navegador
            try:
                await websocket.send_json(resposta_json)

            except (WebSocketDisconnect, RuntimeError) as erro:
                print(f"Conexão WebSocket encerrada durante o envio: {erro}")
                break

            tempo_decode_lm_ms = (t_decode_lm_1 - t_decode_lm_0) * 1000
            tempo_total_ms = (time.perf_counter() - t_inicio_frame) * 1000
            print(
                f"[PERF] decode+landmarks: {tempo_decode_lm_ms:.1f} ms | "
                f"modelo: {tempo_modelo_ms:.1f} ms | "
                f"total_frame: {tempo_total_ms:.1f} ms | "
                f"seq: {len(sequencia)}/30"
            )

    except WebSocketDisconnect:
        print("Cliente desconectado do WebSocket.")