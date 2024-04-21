import os
import torch
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import numpy as np
import soundfile as sf
import subprocess

'''
Load the audio file and return the audio data and sample rate

Parameters:
    audio_path (str): Path to the audio file
    sr (int): Sample rate of the audio file

Returns:
    y (np.ndarray): Audio data
    sr (int): Sample rate
'''
def load_audio(audio_path, sr=None):
    y, sr = sf.read(audio_path)
    return y, sr

'''
Extract features from the audio file

Parameters:
    y (np.ndarray): Audio data
    sr (int): Sample rate

Returns:
    features (list): List of extracted features
'''
def extract_features(y, sr):
    features = []

    # Extract features using librosa
    # Fundamental frequency features
    f0, voiced_flag = librosa.piptrack(y=y, sr=sr)
    features.append(np.mean(f0))  # MDVP:Fo(Hz)
    features.append(np.max(f0))  # MDVP:Fhi(Hz)
    features.append(np.min(f0))  # MDVP:Flo(Hz)

    # Jitter features
    jitter = librosa.effects.split(y, top_db=20)
    features.append(librosa.feature.rms(y=y, frame_length=20, hop_length=10).mean())  # MDVP:Jitter(%)
    features.append(np.mean(np.abs(np.diff(f0))))  # MDVP:Jitter(Abs)
    features.append(np.mean(np.diff(f0)))  # MDVP:RAP
    features.append(np.mean(np.abs(np.diff(np.diff(f0)))))  # MDVP:PPQ
    features.append(np.mean(np.abs(np.diff(f0))) * 3)  # Jitter:DDP

    # Shimmer features
    shimmer = librosa.effects.split(y, top_db=40)
    if shimmer.size > 0:
        shimmer_rms = [librosa.feature.rms(y=y[start:end], frame_length=20, hop_length=10).mean() for start, end in shimmer]
        features.append(np.mean(shimmer_rms))  # MDVP:Shimmer
        # Compute shimmer in dB
        shimmer_db = [np.mean(librosa.amplitude_to_db(librosa.feature.rms(y=y[start:end], frame_length=20, hop_length=10))) for start, end in shimmer]
        features.append(np.mean(shimmer_db))  # MDVP:Shimmer(dB)
        # Compute the mean second-order difference of the signal amplitude as a proxy for sharpness
        sharpness = [np.mean(np.diff(np.abs(np.diff(y[start:end])))) for start, end in shimmer]
        features.append(np.mean(sharpness))  # Shimmer:APQ
        features.append(np.mean(sharpness))  # Shimmer:APQ3
        features.append(np.mean(sharpness))  # Shimmer:APQ5
    else:
        features.extend([0] * 3)  # If shimmer is empty, set to 0 for all shimmer-related features

    # Other features
    features.append(np.mean(librosa.effects.split(y, top_db=60)))  # Shimmer:DDA
    features.append(len(jitter))  # NHR
    features.append(np.mean(librosa.effects.split(y, top_db=60)))  # RPDE
    features.append(np.mean(librosa.effects.split(y, top_db=60)))  # DFA
    features.append(np.mean(librosa.effects.split(y, top_db=60)))  # spread1
    features.append(np.mean(librosa.effects.split(y, top_db=60)))  # spread2
    features.append(np.mean(librosa.effects.split(y, top_db=60)))  # D2
    features.append(np.mean(librosa.effects.split(y, top_db=60)))  # PPE
    features.append(np.mean(librosa.effects.split(y, top_db=60)))  # HNR

    return features

'''
Get the user's voice memo and extract features

Parameters:
    input_file (str): Path to the user's voice memo

Returns:
    features (list): List of extracted features
'''
def get_user_voice_memo(input_file):

    # Check if the input file is an MP3
    if os.path.splitext(input_file)[1].lower() != '.mp3':
        # If not, convert it to MP3
        subprocess.run(['ffmpeg', '-i', input_file, '-acodec', 'libmp3lame', '-q:a', '2', 'output.mp3'])

        # Save path to the converted file
        input_file = '/content/output.mp3'

    y, sr = load_audio(input_file)
    features = extract_features(y, sr)

    return features

'''
Load the saved model and return the model

Parameters:
    model_path (str): Path to the saved model

Returns:
    voice_model (tensorflow.keras.Model): The saved model
'''
def load_voice_model(model_path='./voice_memo_model.h5'):
    # Load the saved model
    return load_model(model_path)

'''
Make a prediction on the user's voice memo

Parameters:
    voice_memo_path (str): Path to the user's voice memo
    
Returns:
    voice_memo_predictions (numpy.ndarray): Prediction probabilities
'''
def make_prediction(voice_memo_path):
    # Get the user's voice memo features
    voice_memo_features = get_user_voice_memo(voice_memo_path)

    # Load the voice memo model
    voice_model = load_voice_model()

    # Make a prediction
    voice_memo_predictions = voice_model.predict([voice_memo_features])

    return voice_memo_predictions

if __name__ == '__main__':
    make_prediction('/content/voice_memo.mp3')