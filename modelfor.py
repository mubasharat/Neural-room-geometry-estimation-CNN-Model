import wave
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU
)
from tensorflow.keras.optimizers import Adam
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import ast  # For parsing rir_metadata
import sys

#np.set_printoptions(threshold=sys.maxsize)

# Function to extract labels from rir_metadata
def extract_metadata(metadata_str):
    """Extract RoomLength, RoomWidth, and RoomHeight from rir_metadata column."""
    try:
        metadata_dict = ast.literal_eval(metadata_str)  # Convert string to dictionary
        return (
            float(metadata_dict.get("RoomLength", 0)),
            float(metadata_dict.get("RoomWidth", 0)),
            float(metadata_dict.get("RoomHeight", 0))
        )
    except (ValueError, SyntaxError):
        return (None, None, None)  # Return None if parsing fails
    
# Load dataset from CSV
def load_dataset(frame_folder, csv_file):
    df = pd.read_csv(csv_file)

    # Extract labels from rir_metadata
    df[['length', 'width', 'height']] = df['rir_metadata'].apply(lambda x: pd.Series(extract_metadata(str(x))))

    frame_files, labels = [], []

    for _, row in df.iterrows():
        audio_folder = os.path.join(frame_folder, os.path.splitext(row['convolved_filename'])[0])

        if os.path.exists(audio_folder):
            for frame_file in os.listdir(audio_folder):
                if frame_file.endswith('.wav'):
                    frame_files.append(os.path.join(audio_folder, frame_file))
                    labels.append((row['length'], row['width'], row['height']))  # Store labels

    return train_test_split(frame_files, labels, test_size=0.2, random_state=42, shuffle=False)

# Preprocess frames for training
def preprocess_frames(frame_files, frame_size):
    data = []
    for file in frame_files:
        fs, signal = wavfile.read(file)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)  # Convert to mono if stereo
        signal = np.pad(signal, (0, max(0, frame_size - len(signal))), mode='constant')  # Pad if shorter
        # Instead of taking just the first frame_size samples, append every chunk.
        for i in range(0, len(signal), frame_size):
            data.append(signal[i:i+frame_size])
    return np.array(data)

# Define CNN model with Regression Output
def get_cnn_model(input_shape):
    inputs = Input(shape=input_shape, name='Input')
    reshape1 = Reshape((input_shape[0], 1), name='Reshape_1')(inputs)

    # Convolutional layers with batch normalization and LeakyReLU
    x = Conv1D(filters=32, kernel_size=4, strides=4, name="1st_Conv1D")(reshape1)
    x = BatchNormalization(name='1st_Batch_Normalization')(x)
    x = LeakyReLU(alpha=0.1, name='1st_LeakyReLU')(x)

    x = Conv1D(filters=64, kernel_size=4, strides=4, name="2nd_Conv1D")(x)
    x = BatchNormalization(name='2nd_Batch_Normalization')(x)
    x = LeakyReLU(alpha=0.1, name='2nd_LeakyReLU')(x)

    x = Conv1D(filters=128, kernel_size=4, strides=2, name="3rd_Conv1D")(x)
    x = BatchNormalization(name='3rd_Batch_Normalization')(x)
    x = LeakyReLU(alpha=0.1, name='3rd_LeakyReLU')(x)

    x = Conv1D(filters=256, kernel_size=3, strides=2, name="4th_Conv1D")(x)
    x = BatchNormalization(name='4th_Batch_Normalization')(x)
    x = LeakyReLU(alpha=0.1, name='4th_LeakyReLU')(x)
    
    x = Conv1D(filters=512, kernel_size=3, strides=2, name="5th_Conv1D")(x)
    x = BatchNormalization(name='5th_Batch_Normalization')(x)
    x = LeakyReLU(alpha=0.1, name='5th_LeakyReLU')(x)
    
    x = Conv1D(filters=512, kernel_size=3, strides=2, name="6th_Conv1D")(x)
    x = BatchNormalization(name='6th_Batch_Normalization')(x)
    x = LeakyReLU(alpha=0.1, name='6th_LeakyReLU')(x)
    
    x = Conv1D(filters=512, kernel_size=2, strides=2, name="7th_Conv1D")(x)
    x = BatchNormalization(name='7th_Batch_Normalization')(x)
    x = LeakyReLU(alpha=0.1, name='7th_LeakyReLU')(x)
    
    x = Conv1D(filters=512, kernel_size=2, strides=2, name="8th_Conv1D")(x)
    x = BatchNormalization(name='8th_Batch_Normalization')(x)
    x = LeakyReLU(alpha=0.1, name='8th_LeakyReLU')(x)

    x = Flatten()(x)  # Flatten before Dense layers
    
    x = Dense(256, activation='relu', name="1st_Dense")(x)
    x = Dense(128, activation='relu', name="2nd_Dense")(x)
    x = Dense(64, activation='relu', name="3rd_Dense")(x)

    # Output layer with 3 continuous values: Length, Width, Height
    outputs = Dense(3, name="Output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Room_Dimension_Regression_Model")
    return model

# Paths
csv_file = "/home/muba4581/convolve/annotations.csv"
frame_folder = "/scratch/muba4581/Frames"  # Where frames were saved
frame_size = 4096  # Same as used during frame extraction

# Load dataset
train_frames, test_frames, train_labels, test_labels = load_dataset(frame_folder, csv_file)

# Convert file paths into processed numpy arrays
X_train = preprocess_frames(train_frames, frame_size)
X_test = preprocess_frames(test_frames, frame_size)

# Convert labels to numpy arrays
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# Sort labels to avoid order issues
y_train = -np.sort(-y_train)
y_test = -np.sort(-y_test)

# Expand dimensions to match model input (needed for 1D Conv)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Get model and compile
input_shape = (frame_size, 1)  # Shape for Conv1D input (samples, 1 channel)
model = get_cnn_model(input_shape)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0002),
    loss='mse',
    metrics=['mae', 'accuracy']
)

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    verbose=1,
    patience=20,         # Stop 20 epochs after no improvement
    restore_best_weights=True  # Restore best model weights
)

# Train model
model.fit(
    X_train, y_train,
    epochs=200,  # Increased for better learning
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]  # Add early stopping callback
)

# Save the trained model
model.save("/home/muba4581/cnn_model.h5")

print("Model training complete. Saved model to cnn_model.h5")
