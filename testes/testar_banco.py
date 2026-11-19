# Testa se a conexão com o banco de dados está funcionando, verifica se o Python consegue acessar o PostgreSQL corretamente.
from database import consultar_sinais

sinais = consultar_sinais()

print("Sinais cadastrados no banco:\n")

for sinal in sinais:
    print(f"ID: {sinal[0]}")
    print(f"Nome: {sinal[1]}")
    print(f"Texto de voz: {sinal[2]}")
    print("-" * 30)

if not sinais:
    print("Nenhum sinal encontrado.")
