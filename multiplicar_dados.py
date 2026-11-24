import os
import glob
import sys

import numpy as np

PASTA_DADOS = "dados_libras"


def aplicar_variacao(sequencia, rng=None):
    """
    Recebe uma matriz (30, 132) ja normalizada pelos ombros e cria uma variacao
    leve e realista, preservando os valores 0.0 de pontos/frames ausentes.
    """
    if rng is None:
        rng = np.random.default_rng()

    nova_seq = sequencia.copy().astype(np.float32)

    # 1. Variacao temporal leve (-1 a +1 frame) para simular inicio/fim levemente adiantado ou atrasado
    deslocamento_tempo = int(rng.choice([-1, 0, 1]))
    if deslocamento_tempo > 0:
        nova_seq = np.vstack([
            np.repeat(nova_seq[:1], deslocamento_tempo, axis=0),
            nova_seq[:-deslocamento_tempo],
        ])
    elif deslocamento_tempo < 0:
        d = abs(deslocamento_tempo)
        nova_seq = np.vstack([
            nova_seq[d:],
            np.repeat(nova_seq[-1:], d, axis=0),
        ])

    # 2. Ruido gaussiano suave apenas nos pontos presentes (nao altera zeros)
    for i in range(nova_seq.shape[0]):
        frame = nova_seq[i]
        if not np.any(frame):
            continue

        # Mao esquerda (0..62)
        if np.any(frame[0:63]):
            mascara_esq = frame[0:63] != 0
            frame[0:63][mascara_esq] += rng.normal(0.0, 0.003, size=np.sum(mascara_esq))

        # Mao direita (63..125)
        if np.any(frame[63:126]):
            mascara_dir = frame[63:126] != 0
            frame[63:126][mascara_dir] += rng.normal(0.0, 0.003, size=np.sum(mascara_dir))

        # Corpo (126..131)
        if np.any(frame[126:132]):
            mascara_corpo = frame[126:132] != 0
            frame[126:132][mascara_corpo] += rng.normal(0.0, 0.002, size=np.sum(mascara_corpo))

        nova_seq[i] = frame

    return nova_seq.astype(sequencia.dtype, copy=False)


def aumentar_treino(x, y, variacoes, rng=None):
    """Devolve exemplos originais + variacoes apenas para o conjunto de treino."""
    if rng is None:
        rng = np.random.default_rng()

    if len(x) != len(y):
        raise ValueError("x e y precisam ter a mesma quantidade de exemplos.")
    if variacoes < 0:
        raise ValueError("variacoes nao pode ser negativo.")

    lista_x = [x]
    lista_y = [y]

    for _ in range(variacoes):
        x_variado = np.array(
            [aplicar_variacao(sequencia, rng) for sequencia in x],
            dtype=x.dtype,
        )
        lista_x.append(x_variado)
        lista_y.append(y)

    return np.concatenate(lista_x, axis=0), np.concatenate(lista_y, axis=0)


def limpar_arquivos_aug():
    """Apaga os *_aug_*.npy antigos caso existam nas pastas de sinais."""
    arquivos = glob.glob(os.path.join(PASTA_DADOS, "*", "*_aug_*.npy"))
    if not arquivos:
        print("Nenhum arquivo *_aug_* encontrado.")
        return

    resposta = input(
        f"Apagar {len(arquivos)} arquivos *_aug_* de '{PASTA_DADOS}'? "
        "As gravacoes reais NAO serao afetadas (s/n): "
    )
    if resposta.strip().lower() == "s":
        for arquivo in arquivos:
            os.remove(arquivo)
        print(f"{len(arquivos)} arquivo(s) aumentado(s) removido(s).")
    else:
        print("Nada foi apagado.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "limpar":
        limpar_arquivos_aug()
    else:
        print("Este modulo e usado pelo treinar_modelo.py")
        print("Para apagar os arquivos *_aug_* antigos: python multiplicar_dados.py limpar")