import cv2
import numpy as np
import mediapipe as mp
import os
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

gestures = ['hello', 'thanks', 'what', 'is', 'your', 'name', 'account_no', 'balance', 'give', 'password', 'have', 'transfer', 'money',
            ' change', 'go', 'done', 'come', 'fast', 'sit', 'here','bye','me','passbook','verification','address','please','sorry','for','wait','no','yes','cheque','phone_no','form','take','can',
            'problem','okay','not_okay','welcome']
            
num_samples_per_class = 50  
timesteps = 15  

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  

for gesture in gestures:
    gesture_dir = os.path.join(DATA_DIR, gesture)
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)

    print(f'Collecting data for gesture: {gesture}')
    time.sleep(2)  
    sample_num = 0
    collected_frames = []  
    
    while sample_num < num_samples_per_class:
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

                if len(collected_frames) == timesteps:
                    npy_path = os.path.join(gesture_dir, f'{gesture}_{sample_num}.npy')
                    np.save(npy_path, np.array(collected_frames))
                    
                    collected_frames = []  
                    sample_num += 1
                    print(f'Saved sample {sample_num}/{num_samples_per_class} for gesture: {gesture}')

        
        cv2.putText(image, f'{gesture}: {sample_num}/{num_samples_per_class}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Data Collection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
hands.close()
