from preprocess.preprocessing import preprocess_data
from models.target_model import train_target_model
from models.shadow_model import train_shadow_models

if __name__ == "__main__":
    
    data_path = "data/adult_clean .csv"
    target_column = "income"

    # Executa o pré-processamento
    X_train, X_val, X_test, y_train, y_val, y_test, shadow_dataset= preprocess_data(
        file_path=data_path, 
        target_col=target_column,
        apply_smote_option=True,   # Defina True ou False para aplicar ou não SMOTE
        apply_scaling_option=True  # Defina True ou False para aplicar ou não MinMaxScaler
    )
    
    print(f"Conjunto de treino: {X_train.shape}, Conjunto de teste: {X_test.shape}")
    print(f"Conjunto Shadow: {shadow_dataset.shape}")

    # Treinando o modelo alvo
    print("Treinando o modelo alvo...")
    target_model, target_history = train_target_model(X_train, y_train, X_val, y_val, X_test, y_test, epochs=20, batch_size=32)

    # Treinando os modelos shadow
    print("Treinando os modelos shadow...")
    shadow_models, shadow_history = train_shadow_models(shadow_dataset, target_column, num_models=5, epochs=20, batch_size=32)

    # Salvando os modelos treinados
    target_model.save("results/output/target_model.h5")
    for idx, model in enumerate(shadow_models):
        model.save(f"results/output/shadow_model_{idx}.h5")

    print("Modelos treinados e salvos com sucesso!")
