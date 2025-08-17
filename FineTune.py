import tensorflow as tf
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd
import wave
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
input_folder = '/home/khta3517/Workspace/R_Audio'  # Path to input folder containing WAV files
#input_folder = '/home/khta3517/Workspace/1000RIR' 
annotation_file = '/home/khta3517/Workspace/FTData.csv'  # Path to annotation CSV
#annotation_file = '/home/khta3517/Workspace/annotations_1000A.csv'  # Path to annotation CSV
frame_size = 4096  # Size of each frame in samples
frames_array, values_array = load_wav_files_as_frames(input_folder, annotation_file, frame_size)

values_array = -np.sort(-values_array)

frames_array = np.expand_dims(frames_array, axis=-1)

# Load the pre-trained model
model_path = "/home/khta3517/Workspace/cnn_model_finetuned2.h5"  # Update the path if needed
model = load_model(model_path)

# Freeze initial layers
for layer in model.layers[:4]:  # Adjust the number based on experimentation
    layer.trainable = False

# Recompile model with a lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.000009),  # Lower learning rate for fine-tuning
    loss=tf.keras.losses.MeanSquaredError(),  # Explicitly use the MSE loss function
    metrics=[tf.keras.metrics.MeanAbsoluteError()]  # Use built-in metric instead of 'mae'
)

# Split the new dataset
X_train, X_test, y_train, y_test = train_test_split(frames_array, values_array, test_size=0.2, random_state=42, shuffle=True)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fine-tune model
history = model.fit(
    X_train, y_train,
    epochs=50,  # Reduce epochs for fine-tuning
    batch_size=32,  
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)
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
plt.savefig('/home/khta3517/Workspace/F_training_mae.png')

# Plot Mean Squared Error (MSE)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train MSE')
plt.plot(history.history['val_loss'], label='Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Mean Squared Error Over Epochs')
plt.grid()
plt.savefig('/home/khta3517/Workspace/F_training_mse.png')

plt.show() 


# Save the fine-tuned model
fine_tuned_model_path = "/home/khta3517/Workspace/cnn_model_finetuned2.h5"
model.save(fine_tuned_model_path)
print(f"Fine-tuned model saved to {fine_tuned_model_path}")