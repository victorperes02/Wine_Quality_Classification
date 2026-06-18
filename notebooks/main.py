import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================
# CARREGAR BASE
# ==========================

vinhos = pd.read_csv('data/WineQT.csv')

# ==========================
# CRIAR CLASSIFICAÇÃO BINÁRIA
# 1 = Alta Qualidade (>= 7)
# 0 = Baixa/Média Qualidade (< 7)
# ==========================

vinhos['qualidade_binaria'] = np.where(
    vinhos['quality'] >= 7,
    1,
    0
)

# Parte 1 do PDF

# ==========================
# SEPARAR DADOS
# ==========================

caracteristicas = vinhos.drop(
    columns=['quality', 'qualidade_binaria', 'Id']
)

resposta = vinhos['qualidade_binaria']

# ==========================
# TREINO E TESTE
# 80% treino
# 20% teste
# ==========================

caracteristicas_treino, caracteristicas_teste, resposta_treino, resposta_teste = train_test_split(
    caracteristicas,
    resposta,
    test_size=0.3,
    random_state=42
)

# Parte 3 do PDF precisa entender a padronização e o modelo de regressão logística para continuar a partir daqui

# ==========================
# PADRONIZAÇÃO
# ==========================

padronizador = StandardScaler()

caracteristicas_treino = padronizador.fit_transform(
    caracteristicas_treino
)

caracteristicas_teste = padronizador.transform(
    caracteristicas_teste
)

# ==========================
# MODELO
# ==========================

modelo = LogisticRegression(
    max_iter=1000
)

modelo.fit(
    caracteristicas_treino,
    resposta_treino
)

# ==========================
# PREVISÕES
# ==========================

previsoes = modelo.predict(
    caracteristicas_teste
)

# ==========================
# ACURÁCIA
# ==========================

acuracia = accuracy_score(
    resposta_teste,
    previsoes
)

print('\n=== RESULTADO GERAL ===')
print(f'Acurácia: {acuracia:.2%}')

# ==========================
# DETALHE DOS VINHOS TESTADOS
# ==========================

resultado = pd.DataFrame()

resultado['Qualidade Original'] = vinhos.loc[
    resposta_teste.index,
    'quality'
]

resultado['Qualidade Real'] = resposta_teste

resultado['Qualidade Prevista'] = previsoes

resultado['Resultado'] = np.where(
    resultado['Qualidade Real']
    == resultado['Qualidade Prevista'],
    'ACERTOU',
    'ERROU'
)

print('\n=== PRIMEIROS 30 VINHOS TESTADOS ===')
print(resultado.head(30))

# ==========================
# RESUMO
# ==========================

acertos = (
    resultado['Resultado']
    == 'ACERTOU'
).sum()

erros = (
    resultado['Resultado']
    == 'ERROU'
).sum()

print('\n=== RESUMO ===')
print(f'Total de vinhos testados: {len(resultado)}')
print(f'Acertos: {acertos}')
print(f'Erros: {erros}')