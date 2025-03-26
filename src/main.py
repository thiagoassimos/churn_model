from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from modeling import train_xgb_model, evaluate_model
from data_preprocessing import clean_data, load_data, categorical_columns



def main():
    """Função principal do script."""
    
    # Carregando os dados
    df = load_data('Churn.csv')
    
    # Removendo colunas irrelevantes (ajustar conforme necessário)
    clean_data(df)
    categorical_columns(df)
     
    # Atribuindo a variável target
    X = df.drop(columns=['CHURN'])
    y = df['CHURN']

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar e avaliar o modelo XGBoost
    print("Treinando o modelo XGBoost...")

    xgb = train_xgb_model(X_train, y_train)

    evaluate_model(xgb, X_test, y_test)


# Executar o código principal
if __name__ == "__main__":
    main()
