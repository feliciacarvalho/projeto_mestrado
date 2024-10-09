import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc

class PrintDot(Callback):
    """Callback para mostrar progresso durante o treinamento."""
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0: 
            print('')
        print('.', end='')

def build_cnn_1d_model(input_shape):
    """
    Constrói uma CNN 1D para classificação binária.
    
    :param input_shape: Tuple que representa a forma do input para o modelo
    :return: Um modelo Keras Sequential
    """
    model = Sequential([
        Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=2, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid') 
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history):
    """Plota o histórico de perda e acurácia."""
    plt.figure(figsize=(12, 5))

    # Plotar perda de treinamento e validação
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss de Treinamento')
    plt.plot(history.history['val_loss'], label='Loss de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim(0, 1)

    # Plotar acurácia (accuracy) de treinamento e validação
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test):
    """Avaliar o modelo e gerar métricas."""
    y_pred_proba = model.predict(X_test)
    y_pred = np.round(y_pred_proba).flatten()

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Calcular a curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plotar a curva ROC
    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

