import os
import sys
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load dataset
with open('gesture_dataset.pkl', 'rb') as f:
    X, y = pickle.load(f)

# Convert labels to categorical
gestures = ['hello', 'thanks', 'what', 'is', 'your', 'name', 'account_no', 'balance', 'give', 'password', 'have', 'transfer', 'money',
            'change', 'go', 'done', 'come', 'fast', 'sit', 'here']
y = tf.keras.utils.to_categorical(y, num_classes=len(gestures))

# Split dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model
model = load_model('gru_gesture_model.keras')

# Evaluate model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Compute accuracy
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=gestures))
