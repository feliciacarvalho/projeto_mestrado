import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """
    Carrega o dataset do caminho especificado.
    """
    return pd.read_csv(file_path)

### --------------------------------------------------------------------------------------------

def drop_correlated_columns(df, columns_to_drop):
    """
    Descartar colunas altamente correlacionadas.
    """
    return df.drop(columns=columns_to_drop, axis=1)

def encode_categorical_columns(df):
    """
    Codificar variáveis categóricas usando OneHotEncoder.
    """
    categoricals = [column for column in df.columns if df[column].dtype == "O"]
    oh_encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded_categoricals = oh_encoder.fit_transform(df[categoricals])

    # Converte as variáveis categóricas codificadas para um DataFrame
    encoded_categoricals = pd.DataFrame(encoded_categoricals, 
                                        columns=oh_encoder.get_feature_names_out().tolist())

    # Junta os dados codificados com o dataset original e remove as colunas categóricas antigas
    df_encoded = df.join(encoded_categoricals)
    df_encoded.drop(columns=categoricals, inplace=True)

    return df_encoded

### --------------------------------------------------------------------------------------------

def split_dataset(df, target_col):
    """
    Separa o dataset em dois: um para o modelo alvo e outro para o shadow.
    O modelo alvo recebe 15.000 amostras e o shadow recebe o restante.
    """
    df.dropna(inplace=True)
    target_dataset = df.sample(n=30000, replace=False)
    shadow_dataset = df.drop(target_dataset.index)

    return target_dataset, shadow_dataset

### --------------------------------------------------------------------------------------------

def prepare_training_data(dataset, target_col, test_size=0.2, val_size=0.2, random_state=1):
    """
    Separa as variáveis preditoras e a variável alvo e divide os dados em treino, validação e teste.
    Retorna arrays numpy ao invés de DataFrames pandas.
    """
    # Separando as features (X) e a variável target (y)
    X = dataset.drop(columns=[target_col]).values  # Convertendo para numpy array
    y = dataset[target_col].values  # Convertendo para numpy array

    # Dividindo em treino + validação e teste
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, 
                                                                random_state=random_state, stratify=y)

    # Dividindo treino em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, 
                                                      random_state=random_state, stratify=y_train_val)

    return X_train, X_val, X_test, y_train, y_val, y_test

### --------------------------------------------------------------------------------------------

def apply_smote(X_train, y_train):
    """
    Aplica SMOTE para balancear o conjunto de dados de treino.
    """
    smote = SMOTE(sampling_strategy="auto")
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled

### --------------------------------------------------------------------------------------------

def apply_minmax_scaling(X_train, X_val, X_test):
    """
    Aplica MinMaxScaler para normalizar as variáveis preditoras.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled

### --------------------------------------------------------------------------------------------

def preprocess_data(file_path, target_col, apply_smote_option=False, apply_scaling_option=False):
    """
    Função completa de pré-processamento. Executa:
    1. Carregamento dos dados
    2. Descartar colunas correlacionadas
    3. Codificação de variáveis categóricas
    4. Separação dos dados em target e shadow datasets
    5. Divisão em treino/validação/teste
    6. Opcional: Aplicação de SMOTE e MinMaxScaler
    """
    
    df = load_data(file_path)
    print("Dataset carregado!!!")

    # Descartar colunas correlacionadas
    columns_to_drop = ['relationship', 'education']
    df = drop_correlated_columns(df, columns_to_drop)
    print("Colunas correlacionadas deletadas!!!")

    # Codificar variáveis categóricas
    df_encoded = encode_categorical_columns(df)
    print("Variáveis categóricas codificadas!!!")

    # Separar datasets para o modelo alvo e shadow
    target_dataset, shadow_dataset = split_dataset(df_encoded, target_col)
    print("Separando dados do modelo alvo e shadow!!!")

    # Preparar os dados para treinamento, validação e teste do modelo alvo
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_training_data(target_dataset, target_col)
    print("Dados de treinamento preparados!!!")

    # Aplicar SMOTE se selecionado
    if apply_smote_option:
        X_train, y_train = apply_smote(X_train, y_train)
        print("Smote aplicado!!!")

    # Aplicar MinMax se selecionado
    if apply_scaling_option:
        X_train, X_val, X_test = apply_minmax_scaling(X_train, X_val, X_test)
        print("MinMax aplicado!!!")

    return X_train, X_val, X_test, y_train, y_val, y_test, shadow_dataset
