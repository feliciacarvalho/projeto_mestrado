import numpy as np
from sklearn.model_selection import train_test_split
from models.cnn_model import build_shadow_cnn_1d_model, PrintDot, plot_training_history, evaluate_model

def train_shadow_models(shadow_dataset, target_col, num_models=5, epochs=20, batch_size=32):
    """
    Treina múltiplos modelos shadow usando CNN 1D.
    
    :param shadow_dataset: O dataset shadow
    :param target_col: Nome da coluna alvo
    :param num_models: Quantidade de modelos shadow a serem treinados
    :param epochs: Número de épocas para treinamento
    :param batch_size: Tamanho do batch
    :return: Lista de modelos shadow treinados
    """
    shadow_models = []

    for i in range(num_models):
        print(f"\nTreinando modelo shadow {i+1}/{num_models}...")

        # Separando as variáveis preditoras e a variável alvo
        X_shadow = shadow_dataset.drop(columns=[target_col]).values
        y_shadow = shadow_dataset[target_col].values

        # Verifique a quantidade total de dados
        num_total_samples = len(X_shadow)
        print(f"Total de amostras: {num_total_samples}")

        # Os 5 mil registros utilizados no treinamento do modelo alvo
        X_train_val = X_shadow[:5000]
        y_train_val = y_shadow[:5000]

        # Os 2 mil registros restantes para o modelo sombra (não usados no modelo alvo)
        X_train_shadow = X_shadow[5000:7000]
        y_train_shadow = y_shadow[5000:7000]

        # Verifique se temos registros suficientes para o conjunto de teste
        if num_total_samples > 7000:
            X_test = X_shadow[7000:]
            y_test = y_shadow[7000:]
        else:
            print(f"Aviso: Total de amostras ({num_total_samples}) não é suficiente para o conjunto de teste de 3000 registros.")
            X_test = X_shadow[5000:]
            y_test = y_shadow[5000:]

        # Concatenando os 5 mil registros de treino do modelo alvo com os 2 mil para o modelo sombra
        X_train = np.concatenate([X_train_val, X_train_shadow], axis=0)
        y_train = np.concatenate([y_train_val, y_train_shadow], axis=0)

        # Expansão para CNN 1D
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)

        input_shape = (X_train.shape[1], 1)
        model = build_shadow_cnn_1d_model(input_shape)

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        shadow_models.append(model)
        print(f"Modelo shadow {i+1}/{num_models} treinado.")
        plot_training_history(history, model_name=f'shadow_model_{i+1}')
        evaluate_model(model, X_test, y_test, model_name=f'shadow_model_{i+1}')
        print("----------------------------------------------------------------------")

    return shadow_models, history
