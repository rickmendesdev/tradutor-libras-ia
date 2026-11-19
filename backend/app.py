import os
import json
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
    raise RuntimeError("Modelo treinado ou arquivo acoes.json não encontrados na pasta 'modelos/'!")

print("Carregando modelo de IA no servidor...")
model = load_model(MODELO_PATH)

with open(ACOES_PATH, "r", encoding="utf-8") as f:
    ACOES = json.load(f)

print(f"Ações carregadas: {ACOES}")


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
    limiar_confianca = 0.75

    try:
        while True:
            # Recebe os bytes da imagem (frame) enviados pelo front-end via WebSocket
            data = await websocket.receive_bytes()
            
            # Converte os bytes recebidos em um array NumPy para o OpenCV
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            # Extrai os landmarks utilizando o módulo landmarks.py
            landmarks, _ = extrair_landmarks(frame)

            sequencia.append(landmarks)
            sequencia = sequencia[-30:]  # Mantém a janela deslizante de 30 frames

            resposta_json = {
                "sinal": "Aguardando sinal...",
                "confianca": 0.0,
                "texto_voz": ""
            }

            if len(sequencia) == 30:
                dados_entrada = np.expand_dims(np.array(sequencia), axis=0)
                
                # Inferência do modelo
                previsoes = model(dados_entrada, training=False).numpy()[0]
                classe_predita = int(np.argmax(previsoes))
                confianca = float(previsoes[classe_predita])

                if confianca >= limiar_confianca:
                    historico_previsoes.append(classe_predita)
                else:
                    historico_previsoes.append(-1)

                validas = [c for c in historico_previsoes if c != -1]

                if validas:
                    mais_comum, qtd = Counter(validas).most_common(1)[0]

                    if qtd >= 3:
                        sinal = ACOES[mais_comum]
                        confianca_exibida = float(previsoes[mais_comum])

                        if sinal == "parado":
                            ultimo_sinal_registrado = None
                        else:
                            if sinal != ultimo_sinal_registrado:
                                texto_voz = registrar_reconhecimento(sinal)
                                ultimo_sinal_registrado = sinal
                            else:
                                texto_voz = None

                        resposta_json = {
                            "sinal": sinal.upper(),
                            "confianca": round(confianca_exibida * 100, 1),
                            "texto_voz": sinal.upper()
                        }

            # Envia o resultado de volta para o JavaScript no navegador
            await websocket.send_json(resposta_json)

    except WebSocketDisconnect:
        print("Cliente desconectado do WebSocket.")