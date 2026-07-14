# Projet de classificação de Qualidade de Vinhos 

## 📌 Sobre o Projeto
Este projeto tem como objetivo desenvolver um modelo preditivo baseado em Machine Learning para classificar a qualidade de vinhos tintos a partir de suas características físico-químicas. A solução visa apoiar enólogos e diretores de produção na tomada de decisão rápida e padronização da qualidade.

Etapas do Projeto
Conhecimento da base de dados
Análise exploratória dos dados (EDA)
Tratamento dos dados
Criação da variável alvo (alta_qualidade)
Análise da distribuição das variáveis
Identificação de outliers
Análise de correlação
Balanceamento das classes utilizando SMOTE
Padronização das variáveis com StandardScaler
Treinamento dos modelos de Machine Learning
Comparação dos modelos
Interpretação da importância das variáveis

## 📂 Estrutura do Repositório
* `data/`: Contém a base de dados original (`winequality-red.csv`).
* `notebooks/`: Jupyter Notebook com o passo a passo da Análise Exploratória (EDA), pré-processamento e modelagem.
* `results/`: Gráficos gerados durante a análise e avaliação dos modelos.

## 🚀 Como Executar
1. Clone o repositório.
2. Instale as dependências listadas no arquivo `requirements.txt` (`pip install -r requirements.txt`).
3. Execute o notebook localizado na pasta `notebooks/`.

## 🛠️ Tecnologias Utilizadas
* Python
* Pandas, NumPy
* Scikit-Learn (Machine Learning)
* Matplotlib, Seaborn (Visualização de Dados)

* Modelos Utilizados
Foram utilizados dois modelos de classificação, sendo eles:

Regressão Logística
Random Forest

Principais Resultados
O modelo Radom Forest apresentou o melhor desempenho geral entres os modelos avaliados

Acurácia: 88%
ROC AUC: 0,897
Melhor modelo: Ramdom Forest
