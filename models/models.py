from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_nn_model(input_dim):
    """
    Fonction pour créer et compiler un modèle de réseau de neurones simple.
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X, y, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=0 ):
    """
    Fonction pour entrainer le modele.
    """
    hist = model.fit(X, y, 
                validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model , hist

def model_predict(model, X):
    """
    Fonction pour faire des predictions sur des nouvelles données.
    """
    y_pred = model.predict(X).flatten()
    return y_pred