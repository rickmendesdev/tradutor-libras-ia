import os
import glob
import numpy as np

# --- CONFIGURACOES ---
PASTA_DADOS = "dados_libras"  # Local onde ficam as pastas e gravacoes dos sinais.
META_POR_PESSOA = 30  # Total final desejado (reais + multiplicados) por pessoa em cada sinal.

def aplicar_variacao(sequencia):
    """
    Recebe uma matriz (30, 132) real e cria uma variacao humana realista.
    Respeita os valores 0.0 de maos/frames ausentes gerados pelo landmarks.py.
    """
    # Trabalha sobre uma copia para nao modificar o arquivo de gravacao original.
    nova_seq = sequencia.copy()

    # 1. Variacao de Tempo (Simula comecar o sinal um pouquinho antes ou depois)
    deslocamento_tempo = np.random.choice([-2, -1, 0, 1, 2])
    if deslocamento_tempo > 0:
        nova_seq = np.vstack([np.repeat(nova_seq[:1], deslocamento_tempo, axis=0), nova_seq[:-deslocamento_tempo]])
    elif deslocamento_tempo < 0:
        d = abs(deslocamento_tempo)
        nova_seq = np.vstack([nova_seq[d:], np.repeat(nova_seq[-1:], d, axis=0)])

    # Sorteia fatores para toda a sequencia, evitando variacoes bruscas entre frames.
    escala = np.random.uniform(0.93, 1.07)     # Variacao de +-7% na amplitude do braco/mao
    shift_x = np.random.uniform(-0.04, 0.04)   # Variacao de posicao horizontal
    shift_y = np.random.uniform(-0.04, 0.04)   # Variacao de altura da mao

    for f in range(nova_seq.shape[0]):
        frame = nova_seq[f]
        # Se o frame inteiro for zero (corpo nao detectado), nao mexe
        if not np.any(frame):
            continue

        # 2. Aplica Escala (multiplicar 0.0 por escala continua 0.0 automaticamente)
        frame = frame * escala

        # 3. Aplica Deslocamento (Shift) e Ruido APENAS nas partes detectadas (diferentes de zero)
        # Mao Esquerda: indices 0 a 63 (21 pontos * 3 coords: x, y, z)
        if np.any(frame[0:63]):
            frame[0:63:3] += shift_x
            frame[1:63:3] += shift_y
            frame[0:63] += np.random.normal(0, 0.006, 63)

        # Mao Direita: indices 63 a 126 (21 pontos * 3 coords: x, y, z)
        if np.any(frame[63:126]):
            frame[63:126:3] += shift_x
            frame[64:126:3] += shift_y
            frame[63:126] += np.random.normal(0, 0.006, 63)

        # Corpo (Nariz e Cotovelos): indices 126 a 132 (3 pontos * 2 coords: x, y)
        if np.any(frame[126:132]):
            frame[126:132:2] += shift_x * 0.5
            frame[127:132:2] += shift_y * 0.5
            frame[126:132] += np.random.normal(0, 0.004, 6)

        nova_seq[f] = frame

    return nova_seq

def multiplicar():
    """Completa a meta de exemplos de cada pessoa criando variacoes das gravacoes reais."""
    # Nao ha nada a aumentar se a pasta principal ainda nao tiver sido criada.
    if not os.path.exists(PASTA_DADOS):
        print(f"Pasta '{PASTA_DADOS}' nao encontrada. Grave alguns sinais primeiro!")
        return

    # Cada subpasta representa um sinal que pode receber novos exemplos.
    acoes = [d for d in os.listdir(PASTA_DADOS) if os.path.isdir(os.path.join(PASTA_DADOS, d))]
    if not acoes:
        print("Nenhuma pasta de sinal encontrada dentro de 'dados_libras'.")
        return

    total_criados = 0

    for acao in sorted(acoes):
        caminho_acao = os.path.join(PASTA_DADOS, acao)
        todos_arquivos = glob.glob(os.path.join(caminho_acao, "*.npy"))

        # Separa apenas as gravacoes REAIS (ignora arquivos que ja tem '_aug_' no nome)
        arquivos_reais = [f for f in todos_arquivos if "_aug_" not in os.path.basename(f)]

        if not arquivos_reais:
            print(f"[{acao}] Nenhuma gravacao real encontrada. Pulando...")
            continue

        # Descobre quais pessoas gravaram este sinal (ex: 'rick', 'izanagi')
        # O trecho antes do primeiro "_" identifica quem fez cada gravacao.
        pessoas = set(os.path.basename(f).split("_")[0] for f in arquivos_reais)

        for pessoa in pessoas:
            reais_pessoa = [f for f in arquivos_reais if os.path.basename(f).startswith(f"{pessoa}_")]
            aug_existentes = glob.glob(os.path.join(caminho_acao, f"{pessoa}_aug_*.npy"))

            # Conta exemplos originais e aumentados antes de calcular quantos faltam.
            qtd_atual = len(reais_pessoa) + len(aug_existentes)
            faltam = META_POR_PESSOA - qtd_atual

            if faltam <= 0:
                print(f"[{acao} - {pessoa}] Ja possui {qtd_atual} sequencias. OK!")
                continue

            # Carrega as matrizes reais dessa pessoa para usar como molde
            # Carrega os originais para que as novas sequencias sejam variacoes deles.
            matrizes_base = [np.load(f) for f in reais_pessoa]

            for i in range(faltam):
                # Escolhe uma gravacao real revezando entre as disponiveis
                molde = matrizes_base[i % len(matrizes_base)]
                sequencia_variada = aplicar_variacao(molde)

                # Usa um nome separado dos originais para nao confundir os dois tipos.
                indice_aug = len(aug_existentes) + i
                nome_saida = f"{pessoa}_aug_{indice_aug}.npy"
                np.save(os.path.join(caminho_acao, nome_saida), sequencia_variada)
                total_criados += 1

            print(f"[{acao} - {pessoa}] {len(reais_pessoa)} reais -> +{faltam} multiplicadas = {META_POR_PESSOA} prontas!")

    print(f"\nConcluido! {total_criados} novas sequencias geradas com sucesso.")

if __name__ == "__main__":
    # Executa a ampliacao do dataset apenas ao iniciar este arquivo diretamente.
    multiplicar()