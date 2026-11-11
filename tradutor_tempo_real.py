import os
import os
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from landmarks import extrair_landmarks
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

print(f"Sinais reconhecidos pelo modelo: {ACOES}")

# Inicializar a webcam
cap = cv2.VideoCapture(0)

sequencia = []
limiar_confianca = 0.75  # Exige pelo menos 75% de certeza da IA

# Histórico curto para suavizar a resposta e evitar falsos positivos rápidos
historico_previsoes = deque(maxlen=5)

print("\n--- TRADUTOR EM TEMPO REAL ATIVADO (MODO FLUIDO) ---")
print("Pressione 'q' na janela do vídeo para fechar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a webcam.")
        break

    # Extrair os landmarks do frame atual
    landmarks, imagem_anotada = extrair_landmarks(frame)

    # Guardar na sequência temporal (últimos 30 frames / 1 segundo)
    sequencia.append(landmarks)
    sequencia = sequencia[-30:]

    # Quando completarmos 30 frames, a IA faz a predição de forma fluida
    if len(sequencia) == 30:
        dados_entrada = np.expand_dims(np.array(sequencia), axis=0)
        
        # Predição otimizada
        previsoes = model(dados_entrada, training=False).numpy()[0]
        classe_predita = int(np.argmax(previsoes))
        confianca = float(previsoes[classe_predita])

        # Se passar do limiar, guarda no histórico; senão, considera neutro (-1)
        if confianca >= limiar_confianca:
            historico_previsoes.append(classe_predita)
        else:
            historico_previsoes.append(-1)

        # Sistema de votação por maioria simples no histórico recente
        validas = [c for c in historico_previsoes if c != -1]
        
        if validas:
            mais_comum, qtd = Counter(validas).most_common(1)[0]
            # Se pelo menos 3 dos últimos quadros concordarem, exibe na tela com estabilidade
            if qtd >= 3:
                sinal_reconhecido = ACOES[mais_comum].upper()
                confianca_exibida = float(previsoes[mais_comum])
                texto = f"{sinal_reconhecido} ({confianca_exibida * 100:.1f}%)"
                cor = (0, 255, 0)  # Verde: Reconhecido com segurança
            else:
                texto = "Analisando..."
                cor = (0, 165, 255)  # Laranja
        else:
            texto = "Aguardando sinal..."
            cor = (0, 165, 255)

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