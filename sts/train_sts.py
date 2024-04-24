import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Lambda

def organize_data(df):
    # Create a neural network based on the df, where y = PD_or_C column

    # Drop participant ID number
    df = df.drop(columns=['Participant ID number', 'Transition ID', 'Clinical_assessment'])

    # Encode categorical variables
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(df.columns)
    df['PD_or_C'] = label_encoder.fit_transform(df['PD_or_C'])
    df['On_or_Off_medication'] = label_encoder.fit_transform(df['On_or_Off_medication'])
    df['STS_additional_features'] = label_encoder.fit_transform(df['STS_additional_features'])
    df['DBS_state'] = label_encoder.fit_transform(df['DBS_state'])

    # Replace missing values
    df['MDS-UPDRS_score_3.9 _arising_from_chair'].fillna(-1, inplace=True)

    return df

# Define a custom threshold function
def custom_threshold(x, threshold=0.5):
    return tf.where(x < threshold, tf.zeros_like(x), tf.ones_like(x))

# Train the model
def train_model(df):
    # Split data into features and target
    X = df.drop(columns=['PD_or_C'])
    y = df['PD_or_C']

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

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

    # Define callbacks for early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # Train the model with callbacks
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Plot training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    # Save the model
    model.save('sts_model')


if __name__ == '__main__':
    # Set the path to the csv
    path = "/content/SitToStand_human_labels.csv"

    # Read the csv into a dataframe
    df = pd.read_csv(path)

    # Organize the data
    df = organize_data(df)

    # Train the model
    train_model(df)