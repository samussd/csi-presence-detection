
# CSI Presence Detection

Este projeto utiliza sinais CSI (Channel State Information) de uma subportadora de Wi-Fi para classificar a presença ou ausência de pessoas em uma sala. O modelo utiliza janelas de 1 segundo (33 amostras) como entrada para uma rede neural convolucional (CNN).

![alt text](https://github.com/samussd/csi-presence-detection/blob/main/modelo.png?raw=true)

## Estrutura do Projeto

```
csi_presence_detection/
├── main.py             # Script principal de execução e treinamento
├── data_loader.py      # Funções para leitura, extração e pré-processamento dos dados
├── model.py            # Definição e compilação da CNN
├── utils.py            # Visualizações dos dados e resultados
└── dados_reduzido/     # Diretório com os arquivos CSV (não incluído no repositório)
```

## Requisitos

- Python 3.8 ou superior
- Bibliotecas:
  - numpy
  - pandas
  - scikit-learn
  - tensorflow
  - matplotlib

Instale os pacotes necessários com:

```bash
pip install -r requirements.txt
```

Para gerar um `requirements.txt`:

```bash
pip freeze > requirements.txt
```

## Organização dos Dados

O projeto espera os dados em `./dados_reduzido/` com a seguinte estrutura:

```
dados_reduzido/
├── scans_vazio/
│   ├── 001.csv
│   ├── 002.csv
│   └── ...
└── scans_cheio/
    ├── 001/
    │   ├── 1.csv
    │   └── ...
    └── 002/
        └── ...
```

Cada arquivo `.csv` representa cerca de 60 segundos de sinal CSI.

## Como Executar

Execute o projeto com:

```bash
python main.py
```

O script realiza os seguintes passos:

1. Carrega e processa os dados de entrada
2. Aplica normalização e split em treino/teste
3. Treina a rede neural com validação
4. Avalia o desempenho final
5. Exibe gráficos de desempenho e um exemplo de sinal

## Resultados

- Acurácia no conjunto de teste
- Histórico gráfico de acurácia (treino vs validação)
- Exemplo visual de um sinal real de 1 segundo

## Observações

- Apenas a parte real da subportadora 16 é usada
- O modelo aplica `EarlyStopping` para evitar overfitting
- O diretório `dados_reduzido` deve estar na raiz do projeto

## Autor

Projeto adaptado de experimento originalmente executado no Google Colab.
