# Análise e previsão de churn de clientes

Este projeto implementa um modelo de Machine Learning para prever a saída de clientes (churn) de uma instituição financeira. O modelo é baseado em um conjunto de dados contendo informações sobre clientes, como: idade, saldo bancário, número de produtos utilizados e histórico de reclamações, entre outras.

## Estrutura do projeto

```
├── data/ --> Contém os dados originais em CSV
├── notebooks/ --> Notebooks Jupyter que descrevem algumas etapas do processo CRISP-DM
│   ├── data_preparation.ipynb --> Notebook que usa as funções de data_preprocessing
│   ├── data_understanding.ipynb --> Notebook para entender os dados
├── output/figures/ --> Contém os resultados gráficos do modelo   
├── src/ --> Scripts Python que contêm funções para pré-processamento e modelagem
│   ├── data_preprocessing.py --> Funções de pré-processamento dos dados
│   ├── modeling.py --> Treinamento e avaliação do modelo preditivo
│   ├── main.py --> Script principal para executar o pipeline
├── DATA_DICT.md --> Dicionário de dados
├── README.md --> Documentação do projeto
├── requirements.txt --> Dependências do projeto
```

## Como executar o pipeline completo

- Instale as dependências e rode o arquivo `main.py`:

```bash
pip install -r requirements.txt
python src/main.py
```

## Escolhas assumidas

- **Pré-processamento:**
  - Removidas colunas irrelevantes (`ID_CLIENTE`, `NOME`, `LOCALIDADE`).
  - Removida a coluna (`TEM_RECLAMACAO`) pois apresentou uma correlação 1 com a variável `CHURN`. Correlações perfeitas, geralmente podem provocar vazamento de dados durante o treinamento, ou seja, isso significa que a variável já contém a resposta. Sendo assim, mantê-la no treinamento, certamente irá provocar resultados que foram "decorados" pelo modelo, o que não é bom sinal quando pretendemos generalizações.
  - Codificadas variáveis categóricas (`GENERO`, `TIPO_CARTAO`).
  - Optei por não excluir qualquer informação das variáveis que foram mantidas no projeto.
- **Modelagem:**
  - Utilizado **`XGBoostClassifier`**, por sua capacidade de capturar relações não lineares e performar bem com classes desbalanceadas.
- **Validação:**
  - Divisão 80/20 entre treino e teste, respectivamente.
  - Utilizadas métricas de `Precision`, `Recall`, `F1-Score`, `Matriz de Confusão` e `Curva ROC`.


## Resultados e conclusões

A ideia é identificar o maior número possível de clientes tipo churn para que possamos tomar ações preventivas para retê-los. Neste caso, é interessante obter um bom resultado para `Recall`. Se o modelo errar e não identificar um cliente que seria churn (falso negativo), isso pode resultar em perda de clientes. Perder clientes sem identificá-los a tempo pode ser muito custoso, especialmente se há algo que se possa fazer para mantê-los. 


## Métricas e interpretação

| **Métrica**  | **Valor**  | **Interpretação** |
|--------------|-----------|-------------------|
| **Precision** | 0.556 | Apenas **50,7% das previsões de churn estão corretas**. Ou seja, o modelo prevê alguns clientes como churn que, na realidade, não churnam (falsos positivos). |
| **Recall** | 0.738 | O modelo conseguiu **identificar 71,5% dos clientes que realmente 'churnam'**. Isso mostra que o modelo tem **boa sensibilidade** ao capturar a maioria dos churns reais. |
| **F1-Score** | 0.634 | A média harmônica entre Precision e Recall. Como o Recall está maior que a Precision, o modelo prioriza **pegar mais clientes churn, mesmo que gere alguns falsos positivos**. |
| **AUC** | 0.865 | O modelo tem **84,8% de chance** de classificar corretamente um churn acima de um não churn. Isso indica que o modelo consegue distinguir bem entre os dois grupos. |
| **Matriz de confusão** | - | - **Verdadeiros negativos (TN) = 1375**: Clientes previstos como não churn corretamente.  - **Falsos positivos (FP) = 232**: Clientes previstos como churn, mas que na realidade não 'churnaram'. **Falsos negativos (FN) = 103**: Clientes previstos como não churn, mas que na realidade churnaram. - **Verdadeiros positivos (TP) = 290**: Clientes previstos como churn corretamente. |


- **Recall boa**: O modelo está conseguindo detectar a maioria dos clientes que churnam, o que é **bom se a empresa quiser evitar perder clientes**.
- **Precision moderada**: Porém, o modelo **comete muitos falsos positivos**, ou seja, prevê que um cliente vai churn, mas ele na verdade fica.
- **F1-Score moderada**: Como a recall é maior que a Precision, o modelo está priorizando **detectar churns ao invés de evitar falsos alarmes**.
- **AUC bom**: Em geral, o modelo tem **boa capacidade de separação** entre churn e não churn.

## Utilidade do modelo
Se a empresa **quer minimizar a perda de clientes a qualquer custo** e pode arcar com ações de retenção mesmo para alguns clientes que não churnariam, **esse modelo está adequado**.

## Sugestão de como melhorar a `precision` (reduzir falsos positivos)
- Ajustar o threshold para ser mais conservador, por exemplo, aumentar de 0.5 para 0.6 ou 0.7.
