# Testa o registro dos sinais reconhecidos no histórico e verifica se os dados estão sendo salvos corretamente no PostgreSQL.
from database import registrar_historico

sinal_id = 2  
if registrar_historico(sinal_id):
    print("Registro inserido no histórico com sucesso!")
else:
    print("Não foi possível registrar o histórico.")
