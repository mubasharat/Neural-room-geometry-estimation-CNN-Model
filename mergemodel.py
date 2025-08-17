import os
import wave
import ast
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Reshape, Flatten, BatchNormalization,
    LeakyReLU, Dropout, GaussianNoise
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# === Load WAV and Annotations from CSV ===
def load_audio_and_labels(input_folder, annotation_file, frame_size):
    annotations = pd.read_csv(annotation_file)

    all_frames, all_labels = [], []

    for _, row in annotations.iterrows():
        filename = row['convolved_filename'].strip()
        filepath = os.path.join(input_folder, filename)

        if not os.path.exists(filepath):
            continue

        try:
            metadata = ast.literal_eval(row['rir_metadata'])
            length = float(metadata.get('RoomLength', 0))
            width = float(metadata.get('RoomWidth', 0))
            height = float(metadata.get('RoomHeight', 0))
        except Exception as e:
            print(f"❌ Skipping row due to metadata parsing error: {e}")
            continue

        try:
            with wave.open(filepath, 'rb') as wav_file:
                n_frames = wav_file.getnframes()
                audio_data = wav_file.readframes(n_frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                total_frames = len(audio_array) // frame_size
                for i in range(total_frames):
                    start = i * frame_size
                    end = start + frame_size
                    frame = audio_array[start:end]
                    if len(frame) == frame_size:
                        all_frames.append(frame)
                        all_labels.append([length, width, height])
        except Exception as e:
            print(f"❌ Skipping file {filename} due to audio read error: {e}")
            continue

    return np.array(all_frames), np.array(all_labels)

# === Label Normalization ===
def normalize_labels(y):
    y_min = y.min(axis=0)
    y_max = y.max(axis=0)
    y_norm = (y - y_min) / (y_max - y_min)
    return y_norm, y_min, y_max

# === Weighted Huber Loss ===
def weighted_huber_loss(y_true, y_pred):
    weights = tf.constant([0.3, 0.3, 0.4])
    error = y_true - y_pred
    delta = 1.0
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * tf.square(quadratic) + delta * linear
    return tf.reduce_mean(loss * weights)

# === Learning Rate Scheduler (fixed) ===
def lr_schedule(epoch):
    if epoch < 10:
        return float(learning_rate * (epoch + 1) / 10.0)  # warmup
    else:
        return float(learning_rate * tf.math.exp(0.03 * (10 - epoch)))

# === CNN Model ===
def get_cnn_model(input_shape):
    inputs = Input(shape=input_shape, name='Input')
    x = GaussianNoise(0.02)(inputs)
    x = Reshape((input_shape[0], 1))(x)

    for filters in [32, 64, 128, 256, 512]:
        x = Conv1D(filters, kernel_size=3, strides=2, padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.4)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(3, name='Output')(x)
    return Model(inputs, outputs)

# === Parameters ===
input_folder = '/home/muba4581/folder_merged'
annotation_file = '/home/muba4581/annotations_merged.csv'
frame_size = 4096
batch_size = 32
epochs = 1000
learning_rate = 0.0001

# === Load and Preprocess ===
X, y = load_audio_and_labels(input_folder, annotation_file, frame_size)
print(f"✅ Loaded {len(X)} audio frames and {len(y)} label entries.")

y, y_min, y_max = normalize_labels(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# === Model ===
model = get_cnn_model((frame_size, 1))
model.compile(optimizer=Adam(learning_rate),
              loss=weighted_huber_loss,
              metrics=['mae'])

# === Callbacks ===
early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
lr_callback = LearningRateScheduler(lr_schedule)

# === Train ===
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, lr_callback],
    verbose=1
)

# === Plot Training Metrics ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Weighted Huber Loss')
plt.plot(history.history['val_loss'], label='Val Weighted Huber Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Mean Absolute Error Over Epochs')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('training_metrics_optimized.png')
plt.show()

# === Save Model and Normalization Params ===
model.save("/home/muba4581/cnn_model_optimized.h5")
np.savez("/home/muba4581/label_norm_params.npz", y_min=y_min, y_max=y_max)

print("✅ Training complete. Model and normalization params saved.")

