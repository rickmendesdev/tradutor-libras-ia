# Bibliotecas para localizar arquivos, salvar configuracoes e manipular matrizes.
import os
import glob
import json
import numpy as np
# Biblioteca usada para criar os graficos do historico de treinamento.
import matplotlib.pyplot as plt
# Ferramentas para dividir os dados e medir o desempenho do classificador.
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
# TensorFlow/Keras fornece as camadas, o treinamento e os callbacks da rede neural.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Mantem o formato dos dados e a pasta do modelo iguais aos usados na coleta.
from landmarks import N_FEATURES, PASTA_MODELOS

# --- 1. CONFIGURACOES ---
# Pasta com subpastas e arquivos .npy de cada sinal.
PASTA_DADOS = "dados_libras"
TAMANHO_SEQUENCIA = 30  # Quantidade de frames esperada em cada exemplo.
# Arquivo da rede treinada.
CAMINHO_MODELO = os.path.join(PASTA_MODELOS, "modelo_libras.keras")
# Ordem dos nomes correspondente as classes do modelo.
CAMINHO_ACOES = os.path.join(PASTA_MODELOS, "acoes.json")
# Imagem com as curvas de treino e validacao.
CAMINHO_GRAFICO = os.path.join(PASTA_MODELOS, "grafico_treino.png")


def carregar_dataset():
    """Carrega sequencias validas e cria os rotulos numericos das acoes."""
    # Sem esta pasta nao ha exemplos para montar o conjunto de treinamento.
    if not os.path.exists(PASTA_DADOS):
        raise SystemExit(
            f"Erro: A pasta '{PASTA_DADOS}' nao existe. Grave os sinais primeiro!")

    # Detecta automaticamente quais pastas de sinais ja possuem gravacoes .npy
    pastas = sorted([
        d for d in os.listdir(PASTA_DADOS)
        if os.path.isdir(os.path.join(PASTA_DADOS, d))
    ])

    # Nomes de sinais que tem dados suficientes para treinar.
    acoes_validas = []
    sequencias = []  # Matrizes de frames aceitas, uma matriz por gravacao.
    rotulos = []  # Indice da acao correspondente a cada sequencia.

    print("\n--- CARREGANDO DADOS ---")
    for acao in pastas:
        arquivos = glob.glob(os.path.join(PASTA_DADOS, acao, "*.npy"))
        if len(arquivos) < 2:
            print(
                f"[AVISO] Sinal '{acao}' ignorado (possui apenas {len(arquivos)} arquivo).")
            continue

        # O indice atribuido aqui sera a classe numerica usada pela rede.
        indice_acao = len(acoes_validas)
        acoes_validas.append(acao)
        contagem = 0

        for arq in arquivos:
            # Cada arquivo deve ter 30 frames, cada um com N_FEATURES coordenadas.
            dados = np.load(arq)
            if dados.shape == (TAMANHO_SEQUENCIA, N_FEATURES):
                sequencias.append(dados)
                rotulos.append(indice_acao)
                contagem += 1
            else:
                print(
                    f"[AVISO] Arquivo ignorado por formato invalido {dados.shape}: {arq}")

        print(f" -> Sinal '{acao}': {contagem} sequencias carregadas.")

    if len(acoes_validas) < 2:
        raise SystemExit(
            "\nErro: Voce precisa ter pelo menos 2 sinais gravados para treinar a IA!")

    # X contem os movimentos; y converte cada classe em um vetor one-hot para a classificacao.
    X = np.array(sequencias, dtype=np.float32)
    y = to_categorical(rotulos, num_classes=len(
        acoes_validas)).astype(np.float32)

    return X, y, acoes_validas


def construir_modelo(num_classes):
    """Cria a Rede Neural Temporal (LSTM) para analisar o movimento nos 30 frames."""
    modelo = Sequential([
        # Define a entrada como 30 frames, cada um com 132 caracteristicas.
        Input(shape=(TAMANHO_SEQUENCIA, N_FEATURES)),

        # Primeira LSTM analisa a sequencia e preserva a saida de cada frame.
        LSTM(64, return_sequences=True),
        # Normaliza as ativacoes para estabilizar o aprendizado e reduz dependencia de neuronios.
        BatchNormalization(),
        Dropout(0.2),

        # Segunda LSTM resume a sequencia temporal em uma representacao do movimento.
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),

        # Camadas densas combinam as caracteristicas para distinguir os sinais.
        Dense(64, activation='relu'),
        Dropout(0.2),

        Dense(32, activation='relu'),
        # Softmax produz uma probabilidade para cada uma das acoes cadastradas.
        Dense(num_classes, activation='softmax')
    ])

    # Usa Adam para ajustar os pesos e entropia cruzada para comparar classes one-hot.
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return modelo


def salvar_grafico(historico):
    """Gera um grafico de precisao e erro para usar nos slides da apresentacao."""
    plt.figure(figsize=(10, 4))

    # Compara a acuracia nos dados usados para aprender e nos dados de validacao.
    plt.subplot(1, 2, 1)
    plt.plot(historico.history['categorical_accuracy'], label='Treino')
    plt.plot(
        historico.history['val_categorical_accuracy'], label='Validacao (Teste)')
    plt.title('Precisao do Modelo (Acuracia)')
    plt.xlabel('Epocas')
    plt.ylabel('Acuracia')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Compara o erro calculado em cada epoca; valores menores indicam previsoes melhores.
    plt.subplot(1, 2, 2)
    plt.plot(historico.history['loss'], label='Treino')
    plt.plot(historico.history['val_loss'], label='Validacao (Teste)')
    plt.title('Taxa de Erro (Loss)')
    plt.xlabel('Epocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Ajusta o espaco entre os paineis, salva a imagem e libera a figura da memoria.
    plt.tight_layout()
    plt.savefig(CAMINHO_GRAFICO, dpi=150)
    plt.close()
    print(f"Grafico de desempenho salvo em: {CAMINHO_GRAFICO}")


def treinar():
    """Executa o carregamento, treinamento, avaliacao e salvamento do modelo."""
    # Cria a pasta de destino se ela ainda nao existir.
    os.makedirs(PASTA_MODELOS, exist_ok=True)

    # 1. Carrega as matrizes (30, 132) e os nomes dos sinais.
    X, y, acoes = carregar_dataset()
    print(
        f"\nTotal carregado: {X.shape[0]} videos | Formato: {X.shape[1]} frames x {X.shape[2]} pontos")

    # 2. Separa 80% para treinamento e 20% para validacao, mantendo a proporcao por classe.
    indices_classes = np.argmax(y, axis=1)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=indices_classes
    )

    # 3. Cria a rede com uma classe de saida para cada sinal e exibe sua arquitetura.
    modelo = construir_modelo(len(acoes))
    modelo.summary()

    # Interrompe quando a validacao deixa de melhorar, reduz a taxa de aprendizado
    # quando o progresso estagna e grava o melhor modelo observado.
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=8, min_lr=1e-5, verbose=1),
        ModelCheckpoint(
            CAMINHO_MODELO, monitor='val_categorical_accuracy', save_best_only=True, verbose=0)
    ]

    # 4. Treina com os exemplos separados e acompanha o desempenho na validacao.
    print("\n--- INICIANDO TREINAMENTO DA REDE NEURAL ---")
    historico = modelo.fit(
        X_treino, y_treino,
        validation_data=(X_teste, y_teste),
        epochs=120,
        batch_size=16,
        callbacks=callbacks
    )

    # 5. Salva a rede e a lista de classes; a ordem dos nomes mapeia os indices de saida.
    modelo.save(CAMINHO_MODELO)
    with open(CAMINHO_ACOES, "w", encoding="utf-8") as f:
        json.dump(acoes, f, ensure_ascii=False, indent=2)

    # 6. Compara a classe prevista e a classe real para calcular as metricas finais.
    previsoes = np.argmax(modelo.predict(X_teste, verbose=0), axis=1)
    reais = np.argmax(y_teste, axis=1)
    precisao_final = accuracy_score(reais, previsoes)

    print("\n" + "=" * 50)
    print(
        f"TREINAMENTO CONCLUIDO! Precisao no Teste: {precisao_final * 100:.2f}%")
    print("=" * 50)
    print("\nRelatorio detalhado por sinal:")
    print(classification_report(reais, previsoes,
          target_names=acoes, zero_division=0))

    salvar_grafico(historico)
    print(f"Modelo salvo em: {CAMINHO_MODELO}")
    print(f"Lista de sinais salva em: {CAMINHO_ACOES}")


if __name__ == "__main__":
    # Executa o treinamento apenas quando este arquivo e iniciado diretamente.
    treinar()
