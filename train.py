import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
import os
import sys

# Ensure UTF-8 encoding for Windows users
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
gestures = ['hello', 'thanks', 'what', 'is', 'your', 'name', 'account_no', 'balance', 'give', 'password', 'have', 'transfer', 'money',
            ' change', 'go', 'done', 'come', 'fast', 'sit', 'here','bye','me','passbook','verification','address','please','sorry','for','wait','no','yes','cheque','phone_no','form','take','can',
            'problem','okay','not_okay','welcome']

# ✅ Check if dataset exists
dataset_path = "gesture_dataset.pkl"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file '{dataset_path}' not found! Ensure it's created correctly.")

# ✅ Load dataset with error handling
with open(dataset_path, "rb") as f:
    try:
        X, y = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

# ✅ Verify dataset structure
if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
    raise ValueError("Dataset format is incorrect. Expected (X, y) as NumPy arrays.")

if X.ndim != 3 or y.ndim != 1:
    raise ValueError(f"Unexpected dataset shape: X={X.shape}, y={y.shape}. Ensure X is (samples, timesteps, features) and y is (samples,).")

samples, timesteps, num_features = X.shape

if samples == 0:
    raise ValueError("Dataset is empty! Ensure it contains valid training samples.")

print(f"✅ Loaded dataset: X shape={X.shape}, y shape={y.shape}")

# ✅ Convert labels to categorical
y = tf.keras.utils.to_categorical(y, num_classes=len(gestures))

# ✅ Define GRU Model
def create_gru_model():
    model = Sequential([
        Input(shape=(timesteps, num_features)),
        GRU(128, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(len(gestures), activation='softmax')
    ])
    return model

gru_model = create_gru_model()
gru_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train Model
history_gru = gru_model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2)

# ✅ Save Model
gru_model.save('gru_gesture_model.keras')  # Use `.keras` instead of `.h5`

print("✅ GRU Model trained and saved as 'gru_gesture_model.keras'.")
