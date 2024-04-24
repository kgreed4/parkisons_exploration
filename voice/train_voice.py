import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Organize the data
def organize_data(X, y):
    # Combine X and y
    y = pd.DataFrame(y)
    df = pd.concat([X, y], axis=1)
    return df

def train_SVC_model(df):
    # Split data into features and target
    X = df.drop(columns=['PD_or_C'])
    y = df['PD_or_C']

    # Split data into training, val, and testing sets into a 70, 15, 15 split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.15, random_state=42)

    # Define the model
    from sklearn.svm import SVC
    model = SVC(kernel='linear', C=1.0, random_state=0)

    # Fit the model
    model.fit(X_train, y_train)

    # Evaluate the model
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    test_score = model.score(X_test, y_test)

    print(f'Training score: {train_score}')
    print(f'Validation score: {val_score}')
    print(f'Test score: {test_score}')

    return model

def train_NN_model(df):
    # Split data into features and target
    X = df.drop(columns=['PD_or_C'])
    y = df['PD_or_C']

    # Split data into training, val, and testing sets into a 70, 15, 15 split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.15, random_state=42)

    # Define the neural network architecture
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

    # Evaluate the model
    train_score = model.evaluate(X_train, y_train)
    val_score = model.evaluate(X_val, y_val)
    test_score = model.evaluate(X_test, y_test)

    print(f'Training score: {train_score}')
    print(f'Validation score: {val_score}')
    print(f'Test score: {test_score}')

    return model

if __name__ == '__main__':
    # Load the dataset
    parkinsons = fetch_ucirepo(id=174)

    # data (as pandas dataframes)
    X = parkinsons.data.features
    y = parkinsons.data.targets

    # Organize the data
    df = organize_data(X, y)

    # Train the classical model
    SVC_model = train_SVC_model(df)

    # Train the neural network model
    NN_model = train_NN_model(df)

    # Save the models
    SVC_model.save('voice_svc_model')
    NN_model.save('voice_nn_model')