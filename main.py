from preprocess.preprocessing import preprocess_data

if __name__ == "__main__":
    # Caminho para o arquivo CSV de dados
    data_path = "data/adult_clean.csv"
    
    # Coluna alvo
    target_column = "income"

    # Executa o pré-processamento
    X_train, X_test, y_train, y_test, shadow_dataset = preprocess_data(
        file_path=data_path, 
        target_col=target_column,
        apply_smote_option=True,   # Defina True para aplicar SMOTE
        apply_scaling_option=True  # Defina True para aplicar MinMaxScaler
    )
    
    
    print(f"Conjunto de treino: {X_train.shape}, Conjunto de teste: {X_test.shape}")
    print(f"Conjunto Shadow: {shadow_dataset.shape}")
