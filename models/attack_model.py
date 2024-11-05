import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def prepare_attack_data(target_data, shadow_data):
    """
    Prepara os dados para o modelo de ataque, dividindo os conjuntos do modelo alvo e shadow.
    Usa 50% do target_data e 70% do shadow_data para treino e teste do modelo de ataque, 
    e o restante para validação.
    """
    
    # Seleciona 50% do target_data para treino/teste e o restante para validação
    num_target_train = int(0.5 * len(target_data))
    target_train_test = target_data[:num_target_train]
    target_validation = target_data[num_target_train:]

    # Seleciona 70% do shadow_data para treino/teste e o restante para validação
    num_shadow_train = int(0.5 * len(shadow_data))
    shadow_train_test = shadow_data[:num_shadow_train]
    shadow_validation = shadow_data[num_shadow_train:]
    
    # Combina os dados de target e shadow para o treino/teste e validação do modelo de ataque
    attack_train_data = np.vstack([target_train_test, shadow_train_test])
    attack_val_data = np.vstack([target_validation, shadow_validation])
    
    # Divide entre preditores e labels para treino/teste e validação
    X_train_attack, y_train_attack = attack_train_data[:, :-1], attack_train_data[:, -1]
    X_val_attack, y_val_attack = attack_val_data[:, :-1], attack_val_data[:, -1]

    return (X_train_attack, y_train_attack), (X_val_attack, y_val_attack)

def define_attack_model(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Usar 'sigmoid' para um problema binário
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_attack_model(y_true, y_pred):
    # Implementar a avaliação do modelo de ataque (pode usar accuracy, precision, recall, etc.)
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
