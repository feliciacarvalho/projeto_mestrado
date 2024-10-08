import numpy as np
from sklearn.model_selection import train_test_split
from models.cnn_model import build_cnn_1d_model

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

        # Dividindo os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X_shadow, y_shadow, test_size=0.25, 
                                                            random_state=42 + i, stratify=y_shadow)

        # Expandindo as dimensões para a CNN 1D
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)

        input_shape = (X_train.shape[1], 1)
        model = build_cnn_1d_model(input_shape)

        # Treinando o modelo
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                            validation_data=(X_test, y_test), verbose=0)

        shadow_models.append(model)
        print(f"Modelo shadow {i+1}/{num_models} treinado.")

    return shadow_models, history
