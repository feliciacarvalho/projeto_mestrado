import numpy as np
from models.cnn_model import build_cnn_1d_model, PrintDot, plot_training_history, evaluate_model

def train_target_model(X_train, X_val, X_test, y_train, y_val, y_test, epochs=20, batch_size=32):
    """
    Treina o modelo alvo usando CNN 1D.
    
    :param X_train: Dados de treino
    :param y_train: Labels de treino
    :param X_test: Dados de teste
    :param y_test: Labels de teste
    :param epochs: Número de épocas
    :param batch_size: Tamanho do batch
    :return: O modelo treinado e o histórico do treinamento
    """
    # dimensão do input para se adequar à CNN 1D
    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=2)

    input_shape = (X_train.shape[1], 1)
    model = build_cnn_1d_model(input_shape)

    # Treinando o modelo
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                        validation_data=(X_val, y_val), callbacks=[PrintDot()], verbose=0)

    print("\nTreinamento do modelo alvo concluído!")
    plot_training_history(history)
    evaluate_model(model, X_test, y_test)


    return model, history
