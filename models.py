"""Deep learning models for DDoS intrusion detection.

Models include Dense, Conv1D, and LSTM architectures for network traffic classification.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam


def model_conv1D(lr=1e-4, N=64, inshape=40):
    """
    Conv1D-based model for DDoS detection.
    
    Args:
        lr (float): Learning rate for Adam optimizer. Default: 1e-4
        N (int): Number of filters in convolutional layers. Default: 64
        inshape (int): Input shape (number of features). Default: 40
    
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        Conv1D(N, 3, activation='relu', input_shape=(inshape, 1), padding='same'),
        Conv1D(N, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Dropout(0.25),
        
        Conv1D(N * 2, 3, activation='relu', padding='same'),
        Conv1D(N * 2, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Dropout(0.25),
        
        Flatten(),
        Dense(N * 4, activation='relu'),
        Dropout(0.5),
        Dense(N * 2, activation='relu'),
        Dropout(0.5),
        Dense(12, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def model_dense(lr=1e-4, N=64, inshape=40):
    """
    Fully connected Dense model for DDoS detection.
    
    Args:
        lr (float): Learning rate for Adam optimizer. Default: 1e-4
        N (int): Number of units in dense layers. Default: 64
        inshape (int): Input shape (number of features). Default: 40
    
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        Dense(N, activation='relu', input_shape=(inshape,)),
        Dropout(0.3),
        
        Dense(N * 2, activation='relu'),
        Dropout(0.3),
        
        Dense(N * 2, activation='relu'),
        Dropout(0.3),
        
        Dense(N, activation='relu'),
        Dropout(0.2),
        
        Dense(12, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def model_lstm(lr=1e-4, N=64, inshape=40):
    """
    LSTM-based model for DDoS detection with temporal patterns.
    
    Args:
        lr (float): Learning rate for Adam optimizer. Default: 1e-4
        N (int): Number of LSTM units. Default: 64
        inshape (int): Input shape (number of features). Default: 40
    
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        LSTM(N, return_sequences=True, input_shape=(inshape, 1)),
        Dropout(0.2),
        
        LSTM(N, return_sequences=True),
        Dropout(0.2),
        
        LSTM(N // 2),
        Dropout(0.2),
        
        Dense(N, activation='relu'),
        Dropout(0.3),
        
        Dense(12, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model