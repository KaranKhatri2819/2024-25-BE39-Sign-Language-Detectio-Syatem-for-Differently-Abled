import numpy as np
import os
import pickle  

DATA_DIR = './data'
gestures = ['hello', 'thanks', 'what', 'is', 'your', 'name', 'account_no', 'balance', 'give', 'password', 'have', 'transfer', 'money',
            ' change', 'go', 'done', 'come', 'fast', 'sit', 'here','bye','me','passbook','verification','address','please','sorry','for','wait','no','yes','cheque','phone_no','form','take','can',
            'problem','okay','not_okay','welcome']

def create_dataset(data_dir, gestures):
    all_samples = []
    all_labels = []
    
    for label, gesture in enumerate(gestures):
        gesture_dir = os.path.join(data_dir, gesture)
        
        if not os.path.exists(gesture_dir):
            print(f"Gesture directory {gesture_dir} does not exist.")
            continue
        
        for file in os.listdir(gesture_dir):
            if file.endswith('.npy'):
                file_path = os.path.join(gesture_dir, file)
                sample = np.load(file_path)
                all_samples.append(sample)
                all_labels.append(label)
                
                print(f'Loaded {file} for gesture: {gesture}')
    
    X = np.array(all_samples)
    y = np.array(all_labels)


    return X, y

X, y = create_dataset(DATA_DIR, gestures)

with open('gesture_dataset.pkl', 'wb') as f:
    pickle.dump((X, y), f)

print(f'Dataset created with {len(X)} samples and saved as gesture_dataset.pkl.')
