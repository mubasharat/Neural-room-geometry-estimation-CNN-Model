import wave
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, Dropout, GaussianNoise
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import ast
import sys

# Function to extract labels from rir_metadata
def extract_metadata(metadata_str):
    try:
        metadata_dict = ast.literal_eval(metadata_str)
        return (
            float(metadata_dict.get("RoomLength", 0)),
            float(metadata_dict.get("RoomWidth", 0)),
            float(metadata_dict.get("RoomHeight", 0))
        )
    except (ValueError, SyntaxError):
        return (None, None, None)

# Load dataset from CSV
def load_dataset(frame_folder, csv_file):
    df = pd.read_csv(csv_file)
    df[['length', 'width', 'height']] = df['rir_metadata'].apply(lambda x: pd.Series(extract_metadata(str(x))))
    frame_files, labels = [], []
    for _, row in df.iterrows():
        audio_folder = os.path.join(frame_folder, os.path.splitext(row['convolved_filename'])[0])
        if os.path.exists(audio_folder):
            for frame_file in os.listdir(audio_folder):
                if frame_file.endswith('.wav'):
                    frame_files.append(os.path.join(audio_folder, frame_file))
                    labels.append((row['length'], row['width'], row['height']))
    return train_test_split(frame_files, labels, test_size=0.2, random_state=42, shuffle=True)

# Preprocess frames
def preprocess_frames(frame_files, frame_size):
    data = []
    for file in frame_files:
        fs, signal = wavfile.read(file)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        signal = np.pad(signal, (0, max(0, frame_size - len(signal))), mode='constant')
        data.append(signal[:frame_size])
    return np.array(data)

# Define CNN model
def get_cnn_model(input_shape):
    inputs = Input(shape=input_shape, name='Input')
    x = GaussianNoise(0.02)(inputs)  # Add noise for regularization
    x = Reshape((input_shape[0], 1))(x)

    for filters in [32, 64, 128, 256, 512]:
        x = Conv1D(filters=filters, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(3, name="Output")(x)
    return Model(inputs, outputs)

# Paths
csv_file = "/scratch/muba4581/convolve/annotations.csv"
frame_folder = "/scratch/muba4581/Frames"
frame_size = 4096

# Load dataset
train_frames, test_frames, train_labels, test_labels = load_dataset(frame_folder, csv_file)
X_train = preprocess_frames(train_frames, frame_size)
X_test = preprocess_frames(test_frames, frame_size)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# Expand dims
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Get model and compile
model = get_cnn_model((frame_size, 1))
model.compile(optimizer=Adam(learning_rate=0.0001), loss=Huber(), metrics=['mae','accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

# Train model
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

# Save model
model.save("/home/muba4581/cnn_model.h5")
print("Model training complete. Saved model to cnn_model.h5")
