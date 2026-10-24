import os
import glob
import sys

import numpy as np

# --- CONFIGURACOES ---
# Local onde ficam as pastas e gravacoes dos sinais.
PASTA_DADOS = "dados_libras"
# Total final desejado (reais + multiplicados) por pessoa em cada sinal.
# META_POR_PESSOA = 30


def aplicar_variacao(sequencia, rng=None):
    """
    Recebe uma matriz (30, 132) real e cria uma variacao humana realista.
    Respeita os valores 0.0 de maos/frames ausentes gerados pelo landmarks.py.
    'rng' e um gerador de numeros aleatorios com semente (resultado reproduzivel);
    """
    if rng is None:
        rng = np.random.default_rng()

    # Trabalha sobre uma copia para nao modificar o arquivo de gravacao original.
    nova_seq = sequencia.copy()

    # 1. Variacao de Tempo (Simula comecar o sinal um pouquinho antes ou depois)
    deslocamento_tempo = np.random.choice([-2, -1, 0, 1, 2])
    if deslocamento_tempo > 0:
        nova_seq = np.vstack([np.repeat(nova_seq[:1], deslocamento_tempo, axis=0),
                              nova_seq[:-deslocamento_tempo]])
    elif deslocamento_tempo < 0:
        d = abs(deslocamento_tempo)
        nova_seq = np.vstack(
            [nova_seq[d:], np.repeat(nova_seq[-1:], d, axis=0)])

    # Sorteia fatores para toda a sequencia, evitando variacoes bruscas entre frames.
    # Variacao de +-7% na amplitude do braco/mao
    escala = np.random.uniform(0.93, 1.07)
    shift_x = np.random.uniform(-0.04, 0.04)   # Variacao de posicao horizontal
    shift_y = np.random.uniform(-0.04, 0.04)   # Variacao de altura da mao

    for i in range(nova_seq.shape[0]):
        frame = nova_seq[i].copy()
        # Se o frame inteiro for zero (corpo nao detectado), nao mexe
        if not np.any(frame):
            continue

        # 2. Aplica Escala (multiplicar 0.0 por escala continua 0.0 automaticamente)
        frame = frame * escala

        # aplicacao de conceito para mao esquerda com 63 valores.
        # Se a mao inteira nao aparecer o bloco com zeros será aplicado deslocamente/ruido.
        esquerda = frame[0:63]
        if np.any(frame[0:63]):
            esquerda[0:63:3] += shift_x
            esquerda[1:63:3] += shift_y
            esquerda[0:63] += rng.normal(0, 0.006, 63)

        # Mao Direita: indices 63 a 126 (21 pontos * 3 coords: x, y, z)
        direita = frame[63:126]
        if np.any(frame[63:126]):
            direita[0:63:3] += shift_x
            direita[1:63:3] += shift_y
            direita += rng.normal(0, 0.006, 63)

        # Corpo (Nariz e Cotovelos): indices 126 a 132 (3 pontos * 2 coords: x, y)
        corpo = frame[126:132]
        if np.any(frame[126:132]):
            corpo[0:6:2] += shift_x * 0.5
            corpo[1:6:2] += shift_y * 0.5
            corpo += rng.normal(0, 0.004, 6)

        nova_seq[i] = frame

    return nova_seq.astype(sequencia.dtype, copy=False)


def aumentar_treino(x, y, variacoes, rng):
    """Devolve exemplo originais + variacoes apenas para o conjunto de treino.
    x:(n,frames,132);y:rotulos correspondentes(por exemplo, one-hot)"""
    if rng is None:
        rng = np.random.default_rng()

    if len(x) != len(y):
        raise ValueError("x e Y precisam ter a mesma quantidade de xemplo.")
    if variacoes < 0:
        raise ValueError("variacoes nao pode ser negativao.")

    lista_x = [x]
    lista_y = [y]

    for _ in range(variacoes):
        X_variado = np.array(
            [aplicar_variacao(sequencia, rng) for sequencia in x],
            dtype=x.dtype
        )
        lista_x.append(X_variado)
        lista_y.append(y)

    return np.concatenate(lista_x, axis=0), np.concatenate(lista_y, axis=0)


def limpar_arquivos_aug():
    """Apaga os *_aug_*.npy antigos (gerados pela versao anterior (comentado em baixo))"""

    arquivos = glob.glob(os.path.join(PASTA_DADOS, "*", "*_aug_*.npy"))
    if not arquivos:
        print("Nenhum arquivo *_aug_* encontrado")
        return
    resposta = input(
        f"Apagar {len(arquivos)} arquivos *_aug_* de {PASTA_DADOS}'? As gravacoes reais NÃO serao afetadas (s/n): ")
    if resposta.strip().lower() == "s":
        for arquivo in arquivos:
            os.remove(arquivo)
            print(f"{len(arquivos)} aquivo(s) aumentado(s) removidos(s).")
    else:
        print("Nada foi apagado.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "limpar":
        limpar_arquivos_aug()
    else:
        print("Este modulo é usado pelo treinar_modelo.py")
        print("Para apagar os arquivos *_aug_* antigos: python multplicar_dados.py limpar")


# Bloco comentado, caso a nova lógica não esteja
# interessante para o projeto sinta-se a vontade sinta-se a vontade para desfazer.
# def multiplicar():
   # """Completa a meta de exemplos de cada pessoa criando variacoes das gravacoes reais."""
    # Nao ha nada a aumentar se a pasta principal ainda nao tiver sido criada.
  #  if not os.path.exists(PASTA_DADOS):
  #     print(
   #         f"Pasta '{PASTA_DADOS}' nao encontrada. Grave alguns sinais primeiro!")
  #      return

    # Cada subpasta representa um sinal que pode receber novos exemplos.
  #  acoes = [d for d in os.listdir(PASTA_DADOS) if os.path.isdir(
  #      os.path.join(PASTA_DADOS, d))]
  #  if not acoes:
   #     print("Nenhuma pasta de sinal encontrada dentro de 'dados_libras'.")
    #   return

    # total_criados = 0

   # for acao in sorted(acoes):
   #     caminho_acao = os.path.join(PASTA_DADOS, acao)
   #     todos_arquivos = glob.glob(os.path.join(caminho_acao, "*.npy"))

    # Separa apenas as gravacoes REAIS (ignora arquivos que ja tem '_aug_' no nome)
    # arquivos_reais = [
    #     f for f in todos_arquivos if "_aug_" not in os.path.basename(f)]

    # if not arquivos_reais:
    #      print(f"[{acao}] Nenhuma gravacao real encontrada. Pulando...")
    #        continue

    # Descobre quais pessoas gravaram este sinal (ex: 'rick', 'izanagi')
    # O trecho antes do primeiro "_" identifica quem fez cada gravacao.
    # pessoas = set(os.path.basename(f).split("_")
    #               [0] for f in arquivos_reais)

    #  for pessoa in pessoas:
    #     reais_pessoa = [f for f in arquivos_reais if os.path.basename(
    #        f).startswith(f"{pessoa}_")]
    #   aug_existentes = glob.glob(os.path.join(
    #       caminho_acao, f"{pessoa}_aug_*.npy"))

    # Conta exemplos originais e aumentados antes de calcular quantos faltam.
    #   qtd_atual = len(reais_pessoa) + len(aug_existentes)
    #   faltam = META_POR_PESSOA - qtd_atual

    # if faltam <= 0:
    #     print(f"[{acao} - {pessoa}] Ja possui {qtd_atual} sequencias. OK!")
    #     continue

    # Carrega as matrizes reais dessa pessoa para usar como molde
    # Carrega os originais para que as novas sequencias sejam variacoes deles.
    # matrizes_base = [np.load(f) for f in reais_pessoa]

    # for i in range(faltam):
    # Escolhe uma gravacao real revezando entre as disponiveis
    #    molde = matrizes_base[i % len(matrizes_base)]
    #     sequencia_variada = aplicar_variacao(molde)

    # Usa um nome separado dos originais para nao confundir os dois tipos.
    #     indice_aug = len(aug_existentes) + i
    #    nome_saida = f"{pessoa}_aug_{indice_aug}.npy"
    #    np.save(os.path.join(caminho_acao, nome_saida),
    #             sequencia_variada)
    #    total_criados += 1

    #  print(
    #       f"[{acao} - {pessoa}] {len(reais_pessoa)} reais -> +{faltam} multiplicadas = {META_POR_PESSOA} prontas!")

   # print(
    # f"\nConcluido! {total_criados} novas sequencias geradas com sucesso.")


# if __name__ == "__main__":
 #   # Executa a ampliacao do dataset apenas ao iniciar este arquivo diretamente.
  #  multiplicar()
