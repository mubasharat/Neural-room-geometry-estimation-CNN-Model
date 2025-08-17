import numpy as np
import wave
import tensorflow as tf
from scipy.io import wavfile
from tensorflow.keras.losses import mse
from tensorflow.keras.metrics import mae

# Function to divide a WAV file into frames
def load_wav_file_as_frames(wav_file, frame_size):
    with wave.open(wav_file, 'rb') as wav:
        n_frames = wav.getnframes()
        audio_data = wav.readframes(n_frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Divide into frames
        total_frames = len(audio_array) // frame_size
        frames = []
        for i in range(total_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio_array[start:end]
            frames.append(frame)
            
    return np.array(frames)

# Path to the input WAV file
#input_wav = "E:/Masters/Project/code/M_Audio/convolved_rir_22417_m1_script4_cleanraw.wav"  # Change this to your WAV file path
frame_size = 4096  # Must match the size used in training
#input_wav = "E:/Masters/Project/code/FTData/recorded_room_11.wav"
input_wav = "E:/Masters/Project/code/8khz/convolved_rir_00001_m1_script4_cleanraw.wav"
#input_wav = "E:/Masters/Project/code/C-AudioTest/convolved_rir_22418_f1_script1_cleanraw.wav"
#input_wav = "E:/Masters/Project/code/M_Audio/convolved_rir_22460_m1_script4_cleanraw.wav"
# Load and preprocess frames
frames_array = load_wav_file_as_frames(input_wav, frame_size)

# Reshape to match CNN model input (add channel dimension)
frames_array = np.expand_dims(frames_array, axis=-1)

# Load trained model
model = tf.keras.models.load_model("C:/Users/TURABI TRADERS/.spyder-py3/11th training/cnn_model.h5")  # Change this to your model path
#model = tf.keras.models.load_model("E:/Masters/Project/code/FineTune/cnn_model_finetuned.h5")
#custom_objects = {'mse': mse, 'mae': mae}

# Make predictions
predictions = model.predict(frames_array)

# Compute the average of all frame predictions
average_prediction = np.mean(predictions, axis=0)

# Display the final estimated room dimensions
m1, m2, m3 = average_prediction
print(f"Estimated Room Dimensions:")
print(f"m1 {m1:.2f} meters")
print(f"m2  {m2:.2f} meters")
print(f"m2 {m3:.2f} meters")
