import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import sys
import os
import time

if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load trained model
model = load_model('gru_gesture_model.keras')

# Define gesture classes
gestures = ['hello', 'thanks', 'what', 'is', 'your', 'name', 'account_no', 'balance', 'give', 'password', 'have', 'transfer', 'money',
            ' change', 'go', 'done', 'come', 'fast', 'sit', 'here','bye','me','passbook','verification','address','please','sorry','for','wait','no','yes','cheque','phone_no','form','take','can',
            'problem','okay','not_okay','welcome']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Set sequence length
timesteps = 15  
collected_frames = []
predicted_sentence = []  # List to store predicted words
max_words = 4  # Max words in the sentence before removing the first word
last_prediction_time = time.time()  # To control 3-sec delay

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video.")
        break

    image = cv2.flip(frame, 1)  
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            keypoints = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            keypoints = np.array(keypoints).flatten()  

            collected_frames.append(keypoints)

            # Predict a word every 'timesteps' frames
            if len(collected_frames) == timesteps:
                if time.time() - last_prediction_time >= 3:  # 3-sec delay
                    X_pred = np.array(collected_frames).reshape(1, timesteps, 63)
                    predictions = model.predict(X_pred)
                    predicted_class = np.argmax(predictions, axis=1)[0]
                    predicted_word = gestures[predicted_class]

                    predicted_sentence.append(predicted_word)  # Add to sentence

                    # Keep max words in the sentence
                    if len(predicted_sentence) > max_words:
                        predicted_sentence.pop(0)  # Remove the first word

                    last_prediction_time = time.time()  # Update last prediction time

                collected_frames = []  # Clear frame buffer

    # Display the predicted sentence on screen
    sentence_text = " ".join(predicted_sentence)  # Join words with space
    cv2.putText(image, sentence_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Sentence Prediction', image)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
