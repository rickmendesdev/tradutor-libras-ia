# Tradutor de Libras com Inteligência Artificial

## 1. Sobre o projeto

Este projeto tem como objetivo desenvolver um sistema capaz de reconhecer sinais de Libras por meio de uma câmera, utilizando inteligência artificial, e relacioná-los a textos em português.

O sistema utiliza um modelo de reconhecimento já treinado para identificar os sinais. Também possui integração com um banco de dados PostgreSQL, responsável por armazenar os sinais cadastrados e manter um histórico dos reconhecimentos realizados.

## 2. Estrutura do projeto

Abaixo está uma explicação geral das principais pastas e arquivos encontrados no projeto.

| Pasta/Arquivo | Função |
|---|---|
| `modelos/` | Armazena os arquivos dos modelos de inteligência artificial utilizados pelo projeto. |
| `__pycache__/` | Pasta criada automaticamente pelo Python para guardar arquivos temporários. |
| `coletar_dados.py` | Responsável pela coleta de dados utilizados no projeto. |
| `database.py` | Realiza a conexão com o PostgreSQL e contém as funções de consulta e registro dos sinais. |
| `landmarks.py` | Contém funções relacionadas à identificação e extração de pontos de referência dos movimentos. |
| `multiplicar_dados.py` | Arquivo relacionado à multiplicação ou ampliação dos dados utilizados no projeto. |
| `treinar_modelo.py` | Utilizado na etapa de treinamento do modelo de inteligência artificial. |
| `teste_camera.py` | Permite verificar se a câmera está funcionando corretamente. |
| `tradutor_tempo_real.py` | Executa o reconhecimento dos sinais em tempo real utilizando a câmera e o modelo treinado. |
| `testar_banco.py` | Utilizado para testar a conexão entre Python e PostgreSQL. |
| `testar_historico.py` | Utilizado para testar o registro dos reconhecimentos no histórico. |
| `.env` | Guarda as configurações individuais de acesso ao banco de dados. |
| `.gitignore` | Define quais arquivos e pastas não devem ser enviados ao Git. |

## 3. Pré-requisitos

Antes de executar o projeto, é necessário ter instalado:

- Python 3.12
- PostgreSQL
- Visual Studio Code (ou outro editor de código)
- Git

**Importante:** recomendamos utilizar o Python 3.12 para manter a compatibilidade com as bibliotecas utilizadas no projeto.

## 4. Baixando o projeto

Para baixar o repositório, abra o terminal e execute:

```powershell
git clone https://github.com/rickmendesdev/tradutor-libras-ia.git
```

Entre na pasta do projeto:

```powershell
cd tradutor-libras-ia
```

Abra o projeto no VS Code:

```powershell
code .
```

## 5. Configurando o ambiente Python

Para evitar conflitos entre versões do Python e bibliotecas de outros projetos, utilizamos um ambiente virtual chamado `.venv`.

Crie o ambiente virtual:

```powershell
py -3.12 -m venv .venv
```

Ative o ambiente virtual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

```powershell
.\.venv\Scripts\Activate.ps1
```

Após a ativação, deverá aparecer `(.venv)` no início da linha do terminal.

Atualize o pip:

```powershell
python -m pip install --upgrade pip
```

Instale as bibliotecas utilizadas:

```powershell
python -m pip install tensorflow opencv-python mediapipe numpy "psycopg[binary]" python-dotenv
```

As principais bibliotecas utilizadas são:

- **TensorFlow:** utilizado para trabalhar com modelos de inteligência artificial.
- **OpenCV:** utilizado para acessar e trabalhar com imagens da câmera.
- **MediaPipe:** utilizado para identificar pontos de referência do corpo e das mãos.
- **NumPy:** utilizado para trabalhar com arrays e dados numéricos.
- **Psycopg:** permite a comunicação entre Python e PostgreSQL.
- **Python-dotenv:** permite carregar as configurações do banco armazenadas no arquivo `.env`.

## 6. Acessando o banco de dados PostgreSQL

O projeto utiliza um banco de dados PostgreSQL já configurado, chamado `learning_db`. 

Não é necessário criar um novo banco de dados ou recriar as tabelas. O objetivo é acessar a estrutura que já existe e utilizar as informações cadastradas.

### 6.1. Acessando pelo pgAdmin

# IMPORTANTE: ESSA SERÁ A FORMA DE ACESSO QUANDO O BANCO NÃO FOR LOCALHOST, A EQUIPE AINDA NÃO CRIOU A ESTRUTURA PARA ISSO.

Abra o pgAdmin e conecte-se ao servidor PostgreSQL utilizando os dados de acesso disponibilizados pela equipe.

Após conectar:

1. Localize o banco de dados `learning_db`.
2. Expanda a opção **Schemas**.
3. Acesse **public**.
4. Entre em **Tables**.

As principais tabelas utilizadas são:

**Tabela `sinais`:**

Armazena os sinais cadastrados no projeto, com seus respectivos identificadores, nomes e textos associados.

**Tabela `historico_learning`:**

Armazena os registros dos sinais reconhecidos pelo sistema, incluindo a data e a hora de cada reconhecimento.

Os 10 sinais utilizados nesta etapa são:

- medo
- amor
- felicidade
- inveja
- tristeza
- raiva
- ansiedade
- empatia
- nojo
- vergonha

**Importante:** cada integrante deve utilizar os dados de conexão fornecidos pela equipe. Não crie outro banco nem altere as tabelas existentes sem necessidade.

## 7. Configurando a conexão com o banco

Cada integrante precisa criar seu próprio arquivo `.env` na pasta principal do projeto.

Esse arquivo é individual e contém as informações necessárias para que o Python consiga acessar o PostgreSQL.

Dentro do arquivo, coloque as configurações de acesso:

```env
DB_HOST=host_do_banco
DB_PORT=porta_do_banco
DB_NAME=learning_db
DB_USER=usuario
DB_PASSWORD=sua_senha
```

Os valores de host, porta, usuário e senha devem ser preenchidos de acordo com os dados de acesso disponibilizados pela equipe.

O arquivo `.env` não deve ser compartilhado nem enviado ao GitHub, pois contém informações privadas de conexão.

Cada integrante deve manter seu próprio arquivo `.env` no computador.

## 8. Testando a conexão com o banco

Depois de configurar o PostgreSQL e o arquivo `.env`, execute no terminal:

```powershell
python testar_banco.py
```

Esse arquivo verifica se o Python consegue estabelecer uma conexão com o banco de dados.

Se a conexão estiver funcionando, será apresentada a mensagem de sucesso configurada no teste.

## 9. Testando o histórico

Para verificar se os registros estão sendo armazenados corretamente, execute:

```powershell
python testar_historico.py
```

Esse teste verifica a funcionalidade de registro no histórico.

Para consultar os registros diretamente no pgAdmin, utilize:

```sql
SELECT
    h.id,
    s.nome,
    s.texto_voz,
    h.data_hora
FROM public.historico_learning h
JOIN public.sinais s
    ON h.sinal_id = s.id
ORDER BY h.data_hora DESC;
```

## 10. Testando a câmera

Antes de executar o reconhecimento, é possível verificar se a webcam está funcionando.

No terminal, execute:

```powershell
python teste_camera.py
```

A janela da câmera deverá abrir. Para fechar, clique na janela do vídeo e pressione a tecla `Q`.

## 11. Executando o tradutor em tempo real

Depois de configurar o ambiente, instalar as bibliotecas e preparar o acesso ao banco de dados, execute:

```powershell
python tradutor_tempo_real.py
```

O sistema deverá abrir a câmera e utilizar o modelo treinado para reconhecer os sinais.

Quando um sinal for reconhecido, o sistema consulta o PostgreSQL para encontrar o texto correspondente e registra automaticamente o reconhecimento na tabela `historico_learning`.

Para encerrar a execução, pressione `Q` na janela da câmera.

## 12. Observações importantes

- Sempre ative o ambiente virtual `.venv` antes de executar os arquivos Python.
- Utilize `python` nos comandos do terminal depois de ativar o ambiente virtual.
- Verifique se o PostgreSQL está acessível antes de executar os testes de conexão.
- Confira se os arquivos do modelo estão presentes na pasta `modelos`.
- Não compartilhe o arquivo `.env` nem envie senhas para o GitHub.
- Não crie novamente as tabelas ou cadastre novamente os sinais no banco já existente.
- Caso ocorra algum erro, confira primeiro as versões do Python, as bibliotecas instaladas e as configurações de conexão com o banco.

---

**Projeto:** Tradutor de Libras com Inteligência Artificial  
**Tecnologias:** Python, TensorFlow, OpenCV, MediaPipe e PostgreSQL.