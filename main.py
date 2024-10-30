from preprocess.preprocessing import preprocess_data
from models.target_model import train_target_model
from models.target_model_tuning import tune_target_model
from models.shadow_model import train_shadow_models
from utils.file_utils import limpar_diretorio 

def main():
    data_path = "data/adult_clean .csv" 
    target_column = "income"

    # Executa o pré-processamento
    X_train, X_val, X_test, y_train, y_val, y_test, shadow_dataset = preprocess_data(
        file_path=data_path, 
        target_col=target_column,
        apply_smote_option=True,   # Defina True ou False para aplicar ou não SMOTE
        apply_scaling_option=True  # Defina True ou False para aplicar ou não MinMaxScaler
    )
    
    print(f"Conjunto de treino: {X_train.shape}, Conjunto de validação: {X_val.shape}, Conjunto de teste: {X_test.shape}")
    print(f"Conjunto Shadow: {shadow_dataset.shape}")

    shadow_models = []  

    while True:
        print("\nSelecione a opção desejada:")
        print("1. Treinar modelo alvo sem tuning de hiperparâmetros")
        print("2. Treinar modelo alvo com tuning de hiperparâmetros")
        print("3. Treinar modelos shadow")
        print("4. Sair")

        option = input("Opção: ")

        if option == '1':
            print("Treinando o modelo alvo sem tuning...")
            target_model, target_history = train_target_model(
                X_train, y_train, X_val, y_val, X_test, y_test, epochs=20, batch_size=32
            )

        elif option == '2':
            print("Limpando o diretório de tuning...")
            limpar_diretorio('results/hp_tuning/cnn_tuning/') 

            print("Treinando o modelo alvo com tuning de hiperparâmetros...")
            best_model, history, best_params = tune_target_model(
                X_train, y_train, X_val, y_val, X_test, y_test
            )

        elif option == '3':
            print("Treinando os modelos shadow...")
            shadow_models, shadow_history = train_shadow_models(
                shadow_dataset, target_column, num_models=5, epochs=20, batch_size=32
            )

        elif option == '4':
            print("Saindo do programa...")
            break

        else:
            print("Opção inválida. Por favor, selecione uma opção válida.")

    # Salvando os pesos dos modelos treinados após a seleção
    target_model.save("results/output/target_model.h5")
    if shadow_models: 
        for idx, model in enumerate(shadow_models):
            model.save(f"results/output/shadow_model_{idx}.h5")
        print("Modelos shadow treinados e salvos com sucesso!")

    print("Modelo alvo treinado e salvo com sucesso!")

if __name__ == "__main__":
    main()
