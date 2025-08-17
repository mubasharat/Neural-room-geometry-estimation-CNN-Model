import wave
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, Dropout
)
from tensorflow.keras.optimizers import Adam
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import ast  # For parsing rir_metadata
import sys
import matplotlib.pyplot as plt



def load_wav_files_as_frames(input_folder, annotation_file, frame_size):
    # Load annotation CSV file
    annotations = pd.read_csv(annotation_file)
    
    # Get a list of all WAV files in the input folder
    wav_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    
    if not wav_files:
        print("No WAV files found in the input folder.")
        return None, None
    
    all_frames = []  # List to store all extracted frames
    all_values = []  # List to store corresponding length, width, height values
    
    # Process each WAV file
    for input_file in wav_files:
        input_path = os.path.join(input_folder, input_file)
        input_filename = os.path.splitext(input_file)[0]  # Extract filename without extension
        
        # Find the corresponding row in the annotation file
        annotations.iloc[:, 0] = annotations.iloc[:, 0].str.replace('.wav', '', regex=False).str.strip()
        input_filename = input_filename.strip().lower()
        annotation_row = annotations[annotations.iloc[:, 0].str.lower() == input_filename]
        
        if annotation_row.empty:
            continue  # Skip files that don't have an annotation entry
        
        #length, width, height = annotation_row.iloc[0, 2:5]  # Extract length, width, height
        length, width, height = annotation_row.loc[:, ["RoomLength", "RoomWidth", "RoomHeight"]].values[0]
        
        # Open the input WAV file
        with wave.open(input_path, 'rb') as wav_file:
            n_frames = wav_file.getnframes()
            
            # Read the entire audio data
            audio_data = wav_file.readframes(n_frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Divide audio into frames
            total_frames = len(audio_array) // frame_size
            for i in range(total_frames):
                start = i * frame_size
                end = start + frame_size
                frame_data = audio_array[start:end]
                all_frames.append(frame_data)
                all_values.append([length, width, height])
                
    return np.array(all_frames), np.array(all_values)  # Convert to NumPy arrays for CNN model input

# Usage example
input_folder = 'E:/Masters/Project/code/Convolved_Audio'  # Path to input folder containing WAV files
annotation_file = 'E:/Masters/Project/code/annotations.csv'  # Path to annotation CSV
frame_size = 4096  # Size of each frame in samples
frames_array, values_array = load_wav_files_as_frames(input_folder, annotation_file, frame_size)
    
# Define CNN model with Regression Output
def get_cnn_model(input_shape):
    inputs = Input(shape=input_shape, name='Input')
    x = inputs
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


# Load dataset
X_train, X_test, y_train, y_test = train_test_split(frames_array, values_array, test_size=0.2, random_state=42,shuffle= False)

y_train = -np.sort(-y_train)
y_test = -np.sort(-y_test)

# Expand dimensions to match model input
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


# Get model and compile
input_shape = (frame_size, 1)
model = get_cnn_model(input_shape)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0002),
    loss=tf.keras.losses.MeanSquaredError(),  # Explicitly use the MSE loss function
    metrics=[tf.keras.metrics.MeanAbsoluteError()]  # Use built-in metric instead of 'mae'
   #metrics={'length': 'mae', 'width': 'mae', 'height': 'mae'}
)

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=400,  # Increased for better learning
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]  # Add early stopping callback
    )

# Plot training history
plt.figure(figsize=(12, 5))

print(history.history.keys())

# Plot Mean Absolute Error (MAE)
plt.subplot(1, 2, 1)
plt.plot(history.history['mean_absolute_error'], label='Train MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.title('Mean Absolute Error Over Epochs')
plt.grid()
plt.savefig('mae_plot.png')

# Plot Mean Squared Error (MSE)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train MSE')
plt.plot(history.history['val_loss'], label='Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Mean Squared Error Over Epochs')
plt.grid()
plt.savefig('mse_plot.png')

plt.show()



# Save the trained model
model.save("E:/Masters/Project/code/cnn_model.h5")

print("Model training complete. Saved model to cnn_model.keras")