import tensorflow as tf
import pyaudio
import wave
import os
import numpy as np
import smbus
import time
from tensorflow.keras.losses import mse
from tensorflow.keras.metrics import mae

# Constants
FRAME_SIZE = 4096
FS = 44100  # Sampling rate
CHANNELS = 1
CHUNK = 1024
DURATION = 5  # seconds
AUDIO_FILENAME = "recorded_audio.wav"  # File to save the audio

# LCD Constants
I2C_ADDR = 0x27  # Change if your LCD address is different
LCD_CHR = 1  # Mode - Sending data
LCD_CMD = 0  # Mode - Sending command
LINE_1 = 0x80  # Address for the first line
LINE_2 = 0xC0  # Address for the second line
LCD_BACKLIGHT = 0x08  # LCD backlight on
ENABLE = 0b00000100  # Enable bit

# Initialize I2C bus
bus = smbus.SMBus(1)

def lcd_byte(bits, mode):
    """Send byte to data pins via I2C."""
    high_bits = mode | (bits & 0xF0) | LCD_BACKLIGHT
    low_bits = mode | ((bits << 4) & 0xF0) | LCD_BACKLIGHT
    bus.write_byte(I2C_ADDR, high_bits)
    lcd_toggle_enable(high_bits)
    bus.write_byte(I2C_ADDR, low_bits)
    lcd_toggle_enable(low_bits)

def lcd_toggle_enable(bits):
    """Toggle enable pin."""
    time.sleep(0.0005)
    bus.write_byte(I2C_ADDR, (bits | ENABLE))
    time.sleep(0.0005)
    bus.write_byte(I2C_ADDR, (bits & ~ENABLE))
    time.sleep(0.0005)

def lcd_init():
    """Initialize the LCD."""
    lcd_byte(0x33, LCD_CMD)
    lcd_byte(0x32, LCD_CMD)
    lcd_byte(0x06, LCD_CMD)
    lcd_byte(0x0C, LCD_CMD)
    lcd_byte(0x28, LCD_CMD)
    lcd_byte(0x01, LCD_CMD)
    time.sleep(0.005)

def lcd_string(message, line):
    """Display message on LCD."""
    lcd_byte(line, LCD_CMD)
    for char in message.ljust(16, " "):
        lcd_byte(ord(char), LCD_CHR)

def lcd_clear():
    """Clear the LCD screen."""
    lcd_byte(0x01, LCD_CMD)
    time.sleep(0.005)

# Load the trained model
MODEL_PATH = "cnn_model_finetuned_1.h5"
if not tf.io.gfile.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()

custom_objects = {'mse': mse, 'mae': mae}
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("Model loaded successfully.")

def preprocess_audio(signal, frame_size=FRAME_SIZE):
    """Preprocess raw audio signal into frames for model input."""
    frames = [signal[i * frame_size:(i + 1) * frame_size] for i in range(len(signal) // frame_size)]
    return np.array(frames)

def predict_room_dimensions(model, audio_buffer):
    """Use the model to predict room dimensions."""
    frames = preprocess_audio(audio_buffer)
    if frames.shape[0] == 0:
        print("Error: No valid audio frames processed.")
        return None
    
    predictions = model.predict(frames)
    avg_prediction = np.mean(predictions, axis=0)
    
    results = {
        'm1': float(avg_prediction[0]),
        'm2': float(avg_prediction[1]),
        'm3': float(avg_prediction[2])
    }
    
    # Print results to console
    print(f"\nPredicted Room Dimensions:")
    print(f"m1: {results['m1']:.2f} m")
    print(f"m2: {results['m2']:.2f} m")
    print(f"m3: {results['m3']:.2f} m")
    
    # Display results on LCD
    lcd_string(f"m1:{results['m1']:.2f}m", LINE_1)
    lcd_string(f"m2:{results['m2']:.2f}m, m3:{results['m3']:.2f}m", LINE_2)
    time.sleep(3)
    lcd_clear()
    
    return results

def record_and_predict():
    """Continuously record, save, and process audio in 5-second buffers."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=FS, input=True, frames_per_buffer=CHUNK)
    print("Continuous recording started. Press Ctrl+C to stop.")

    TARGET_LENGTH = 53 * FRAME_SIZE  # Target samples based on training data
    
    try:
        while True:
            audio_buffer = np.concatenate([np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16) for _ in range(int(FS / CHUNK * DURATION))])
            audio_buffer = np.pad(audio_buffer, (0, max(0, TARGET_LENGTH - len(audio_buffer))), mode='constant')[:TARGET_LENGTH]
            
            # Predict room dimensions
            predict_room_dimensions(model, audio_buffer)
    
    except KeyboardInterrupt:
        print("\nStopping recording.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    lcd_init()
    record_and_predict()
