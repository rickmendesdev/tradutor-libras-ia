# Faz a conexão do projeto com o banco de dados PostgreSQL e contém as funções para consultar os sinais e registrar os reconhecimentos no histórico.
import os
#psycopg.connect(): estabelece a conexão com o PostgreSQL
import psycopg
from dotenv import load_dotenv

#load_dotenv(): carrega as configurações do arquivo .env
load_dotenv() 

def conectar():
    try:
        conexao = psycopg.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
#return conexao: devolve a conexão para os outros códigos utilizarem
        return conexao
#except: identifica possíveis erros na conexão
    except psycopg.Error as erro:
        print(f"Erro ao conectar ao PostgreSQL: {erro}")
        return None

#Conectando ao banco
def consultar_sinais():
    conexao = conectar()

    if conexao is None:
        return []
#SELECT na tabela sinais
    try:
        with conexao.cursor() as cursor:
            cursor.execute(
                "SELECT id, nome, texto_voz FROM public.sinais"
            )
#Busca os registros existentes e retorna os dados para o Python
            sinais = cursor.fetchall()
            return sinais

    except psycopg.Error as erro:
        print(f"Erro ao consultar sinais: {erro}")
        return []
#Fecha a conexão ao terminar
    finally:
        conexao.close()


def registrar_historico(sinal_id):
    conexao = conectar()

    if conexao is None:
        return False

    try:
        with conexao.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO public.historico_learning (sinal_id)
                VALUES (%s)
                """,
                (sinal_id,)
            )

        conexao.commit()
        return True

    except psycopg.Error as erro:
        conexao.rollback()
        print(f"Erro ao registrar histórico: {erro}")
        return False

    finally:
        conexao.close()

def registrar_reconhecimento(nome_sinal):
    conexao = conectar()
#recebe o nome reconhecido pela IA, procura esse nome na tabela sinais, registra o ID encontrado no histórico e devolve o texto_voz
    if conexao is None:
        return None

    try:
        with conexao.cursor() as cursor:

            # Procurar o sinal no banco
            cursor.execute(
                """
                SELECT id, texto_voz
                FROM public.sinais
                WHERE LOWER(nome) = LOWER(%s)
                """,
                (nome_sinal,)
            )

            resultado = cursor.fetchone()

            if resultado is None:
                print(f"Sinal '{nome_sinal}' não encontrado no banco.")
                return None

            sinal_id, texto_voz = resultado

            # Registrar o reconhecimento no histórico
            cursor.execute(
                """
                INSERT INTO public.historico_learning (sinal_id)
                VALUES (%s)
                """,
                (sinal_id,)
            )

        conexao.commit()

        return texto_voz

    except psycopg.Error as erro:
        conexao.rollback()
        print(f"Erro ao registrar reconhecimento: {erro}")
        return None

    finally:
        conexao.close()