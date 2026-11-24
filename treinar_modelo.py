import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from backend.landmarks import N_FEATURES, PASTA_MODELOS
from multiplicar_dados import aumentar_treino

PASTA_DADOS = "dados_libras"
TAMANHO_SEQUENCIA = 30
VARIACOES_TREINO = 3
CAMINHO_MODELO = os.path.join(PASTA_MODELOS, "modelo_libras.keras")
CAMINHO_CHECKPOINT = os.path.join(PASTA_MODELOS, "melhor_modelo.keras")
CAMINHO_ACOES = os.path.join(PASTA_MODELOS, "acoes.json")
CAMINHO_GRAFICO = os.path.join(PASTA_MODELOS, "grafico_treino.png")


def carregar_dataset():
    """Carrega sequencias validas e cria os rotulos numericos das acoes."""
    if not os.path.exists(PASTA_DADOS):
        raise SystemExit(
            f"Erro: A pasta '{PASTA_DADOS}' nao existe. Grave os sinais primeiro!"
        )

    pastas = sorted([
        d for d in os.listdir(PASTA_DADOS)
        if os.path.isdir(os.path.join(PASTA_DADOS, d))
    ])

    acoes_validas = []
    sequencias = []
    rotulos = []

    print("\n--- CARREGANDO DADOS ---")
    for acao in pastas:
        todos_arquivos = glob.glob(os.path.join(PASTA_DADOS, acao, "*.npy"))
        arquivos = [f for f in todos_arquivos if "_aug_" not in os.path.basename(f)]

        if len(arquivos) < 2:
            print(
                f"[AVISO] Sinal '{acao}' ignorado (possui apenas {len(arquivos)} arquivo real)."
            )
            continue

        indice_acao = len(acoes_validas)
        acoes_validas.append(acao)
        contagem = 0

        for arq in arquivos:
            dados = np.load(arq)
            if dados.shape == (TAMANHO_SEQUENCIA, N_FEATURES):
                sequencias.append(dados)
                rotulos.append(indice_acao)
                contagem += 1
            else:
                print(
                    f"[AVISO] Arquivo ignorado por formato invalido {dados.shape}: {arq}"
                )

        print(f" -> Sinal '{acao}': {contagem} sequencias reais carregadas.")

    if len(acoes_validas) < 2:
        raise SystemExit(
            "\nErro: Voce precisa ter pelo menos 2 sinais gravados para treinar a IA!"
        )

    X = np.array(sequencias, dtype=np.float32)
    y = to_categorical(rotulos, num_classes=len(acoes_validas)).astype(np.float32)

    return X, y, acoes_validas


def construir_modelo(num_classes):
    """Cria a Rede Neural Temporal (LSTM) para analisar o movimento nos 30 frames."""
    modelo = Sequential([
        Input(shape=(TAMANHO_SEQUENCIA, N_FEATURES)),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return modelo


def salvar_grafico(historico):
    """Gera um grafico de precisao e erro para usar nos slides da apresentacao."""
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(historico.history["categorical_accuracy"], label="Treino")
    plt.plot(
        historico.history["val_categorical_accuracy"], label="Validacao (Teste)"
    )
    plt.title("Precisao do Modelo (Acuracia)")
    plt.xlabel("Epocas")
    plt.ylabel("Acuracia")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(historico.history["loss"], label="Treino")
    plt.plot(historico.history["val_loss"], label="Validacao (Teste)")
    plt.title("Taxa de Erro (Loss)")
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CAMINHO_GRAFICO, dpi=150)
    plt.close()
    print(f"Grafico de desempenho salvo em: {CAMINHO_GRAFICO}")


def treinar():
    """Executa o carregamento, treinamento, avaliacao e salvamento do modelo."""
    os.makedirs(PASTA_MODELOS, exist_ok=True)

    X, y, acoes = carregar_dataset()
    print(
        f"\nTotal real carregado: {X.shape[0]} sequencias | "
        f"Formato: {X.shape[1]} frames x {X.shape[2]} pontos"
    )

    indices_classes = np.argmax(y, axis=1)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=indices_classes
    )

    rng = np.random.default_rng(42)
    X_treino, y_treino = aumentar_treino(
        X_treino, y_treino, variacoes=VARIACOES_TREINO, rng=rng
    )
    print(
        f"Apos multiplicacao em memoria: {X_treino.shape[0]} exemplos de treino | "
        f"{X_teste.shape[0]} exemplos reais de teste"
    )

    modelo = construir_modelo(len(acoes))
    modelo.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-5,
            verbose=1,
        ),
        ModelCheckpoint(
            CAMINHO_CHECKPOINT,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    print("\n--- INICIANDO TREINAMENTO DA REDE NEURAL ---")
    historico = modelo.fit(
        X_treino,
        y_treino,
        validation_data=(X_teste, y_teste),
        epochs=120,
        batch_size=16,
        callbacks=callbacks,
    )

    # Usa o melhor checkpoint salvo se ele existir; caso contrario, mantem os pesos restaurados
    if os.path.exists(CAMINHO_CHECKPOINT):
        modelo_final = load_model(CAMINHO_CHECKPOINT)
    else:
        modelo_final = modelo

    modelo_final.save(CAMINHO_MODELO)
    with open(CAMINHO_ACOES, "w", encoding="utf-8") as f:
        json.dump(acoes, f, ensure_ascii=False, indent=2)

    previsoes = np.argmax(modelo_final.predict(X_teste, verbose=0), axis=1)
    reais = np.argmax(y_teste, axis=1)
    precisao_final = accuracy_score(reais, previsoes)

    print("\n" + "=" * 50)
    print(f"TREINAMENTO CONCLUIDO! Precisao no Teste: {precisao_final * 100:.2f}%")
    print("=" * 50)
    print("\nRelatorio detalhado por sinal:")
    print(classification_report(reais, previsoes, target_names=acoes, zero_division=0))

    salvar_grafico(historico)
    print(f"Modelo salvo em: {CAMINHO_MODELO}")
    print(f"Lista de sinais salva em: {CAMINHO_ACOES}")


if __name__ == "__main__":
    treinar()