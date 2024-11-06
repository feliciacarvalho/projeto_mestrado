import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report

def prepare_attack_data(target_model, shadow_models, target_train_data, target_test_data, shadow_train_data, shadow_test_data):
    """
    Prepara os dados para o modelo de ataque, usando as previsões dos modelos sombra e do modelo alvo.
    
    :param target_model: O modelo alvo treinado
    :param shadow_models: Lista de modelos sombra treinados
    :param target_train_data: Dados de treino do modelo alvo
    :param target_test_data: Dados de teste do modelo alvo
    :param shadow_train_data: Dados de treino dos modelos sombra
    :param shadow_test_data: Dados de teste dos modelos sombra
    :return: Dados de treino e validação para o modelo de ataque
    """
    
    # Gerar as previsões de probabilidade para os dados de treino e teste
    target_train_preds = target_model.predict(target_train_data)
    target_test_preds = target_model.predict(target_test_data)

    # Gerar as previsões dos modelos sombra para os dados de treino e teste
    shadow_train_preds = np.array([model.predict(shadow_train_data) for model in shadow_models])
    shadow_test_preds = np.array([model.predict(shadow_test_data) for model in shadow_models])

    # As previsões do modelo alvo e dos modelos sombra serão usadas como features
    # E as labels serão 1 para membros (modelo alvo) e 0 para não-membros (modelos sombra)
    
    # Dados para treino
    attack_train_data = np.concatenate([shadow_train_preds.T, target_train_preds], axis=0)
    attack_train_labels = np.concatenate([np.zeros(len(shadow_train_preds[0])), np.ones(len(target_train_preds))])

    # Dados para teste
    attack_test_data = np.concatenate([shadow_test_preds.T, target_test_preds], axis=0)
    attack_test_labels = np.concatenate([np.zeros(len(shadow_test_preds[0])), np.ones(len(target_test_preds))])

    return (attack_train_data, attack_train_labels), (attack_test_data, attack_test_labels)


def define_attack_model(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Usar 'sigmoid' para um problema binário
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def evaluate_attack_model(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)


# Função principal
def train_and_evaluate_attack_model(target_model, shadow_models, target_train_data, target_test_data, shadow_train_data, shadow_test_data):
    # Preparar os dados para o modelo de ataque
    (attack_train_data, attack_train_labels), (attack_test_data, attack_test_labels) = prepare_attack_data(
        target_model, shadow_models, target_train_data, target_test_data, shadow_train_data, shadow_test_data
    )

    # Definir o modelo de ataque
    attack_model = define_attack_model(input_dim=attack_train_data.shape[1])

    # Treinamento do modelo de ataque
    attack_model.fit(attack_train_data, attack_train_labels, epochs=10, batch_size=32, validation_data=(attack_test_data, attack_test_labels))

    # Avaliação do modelo de ataque
    y_pred_attack = attack_model.predict(attack_test_data)
    evaluate_attack_model(attack_test_labels, y_pred_attack)
