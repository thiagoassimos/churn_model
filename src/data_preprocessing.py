import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(file_name):
    """Carrega o dataset de um arquivo CSV."""
    path = os.path.join(os.getcwd(), '..', 'data', file_name)
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame):
    """Remove colunas irrelevantes."""
    df.drop(columns=["ID_CLIENTE", "NOME", "TEM_RECLAMACAO"], inplace=True)
    return df
    
def categorical_columns(df: pd.DataFrame):
    """Realiza o pré-processamento de dados."""  
    categorical_columns = ["GENERO", "TIPO_CARTAO", "LOCALIDADE"]
    label_encoders = {} 
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df
