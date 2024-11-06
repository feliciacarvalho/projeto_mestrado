import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
import os

class PrintDot(Callback):
    """Callback para mostrar progresso durante o treinamento."""
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0: 
            print('')
        print('.', end='')

## Função original
# def build_cnn_1d_model(input_shape):
#     model = Sequential([
#         Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape),
#         MaxPooling1D(pool_size=2),
#         Conv1D(128, kernel_size=2, activation='relu'),
#         MaxPooling1D(pool_size=2),
#         Flatten(),
#         Dense(64, activation='relu'),
#         Dense(1, activation='sigmoid') 
#     ])

#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model


## Função após adicionar os melhores parametros encontrados com o tunning
def build_cnn_1d_model(input_shape=(84, 1)): 

    model = Sequential([
        #primeira camada conv
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),  # filters=64, kernel_size=3
        MaxPooling1D(pool_size=2),  

        #segunda camada conv
        Conv1D(128, kernel_size=3, activation='relu'),  # filters2=128, kernel_size2=3
        MaxPooling1D(pool_size=3), 

        Flatten(),

        Dense(96, activation='relu'),  # dense_units=96

        Dense(1, activation='sigmoid') 
    ])

    model.compile(optimizer='rmsprop',  
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    return model


def build_shadow_cnn_1d_model(input_shape):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid para gerar probabilidades para classificação binária
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Loss para binário
    return model


# def plot_training_history(history, model_name):
#     """Plota e salva o histórico de perda e acurácia."""
#     plt.figure(figsize=(12, 5))

#     # Plotar perda de treinamento e validação
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Loss de Treinamento')
#     plt.plot(history.history['val_loss'], label='Loss de Validação')
#     plt.xlabel('Épocas')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.ylim(0, 1)

#     # Plotar acurácia de treinamento e validação
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
#     plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
#     plt.xlabel('Épocas')
#     plt.ylabel('Acurácia')
#     plt.legend()
#     plt.ylim(0, 1)

#     plt.tight_layout()
#     plt.savefig(f'results/metrics/{model_name}_training_history.png') 
#     plt.close()  


def plot_training_history(history, model_name):
    """Plota e salva o histórico de perda e acurácia, verificando a presença de dados de validação."""
    plt.figure(figsize=(12, 5))

    # Plotar perda de treinamento e validação
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss de Treinamento')
    
    # Verificar se 'val_loss' existe antes de plotar
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Loss de Validação')

    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim(0, 1)

    # Plotar acurácia de treinamento e validação
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    
    # Verificar se 'val_accuracy' existe antes de plotar
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')

    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f'results/metrics/{model_name}_training_history.png')
    plt.close()

def evaluate_model(model, X_test, y_test, model_name):
    """Avaliar o modelo e gerar métricas."""
    if X_test.shape[0] == 0:
        print(f"Erro: Dados de teste vazios para o modelo {model_name}.")
        return

    # Realizar a predição
    y_pred_proba = model.predict(X_test)
    
    # Verificar se as predições não estão vazias
    if y_pred_proba.shape[0] == 0:
        print(f"Erro: Predições vazias para o modelo {model_name}.")
        return

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
    plt.savefig(f'results/metrics/{model_name}_roc_curve.png')
    plt.close()

