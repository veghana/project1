import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_fer2013():
    print(f"Current working directory: {os.getcwd()}")
    
    # List all files in the current directory and its parent
    print("Files in current directory:")
    print(os.listdir('.'))
    print("Files in parent directory:")
    print(os.listdir('..'))
    
    # Try multiple possible locations for the CSV file
    possible_locations = [
        'fer2013.csv',
        '../fer2013.csv',
        '../data/fer2013.csv',
        'data/fer2013.csv'
    ]
    
    data = None
    for location in possible_locations:
        try:
            print(f"Attempting to load from: {location}")
            data = pd.read_csv(location)
            print(f"Successfully loaded data from {location}")
            break
        except FileNotFoundError:
            print(f"File not found at {location}")
    
    if data is None:
        raise FileNotFoundError("Could not find fer2013.csv in any expected location")
    
    # Extract features and labels
    pixels = data['pixels'].tolist()
    emotions = pd.get_dummies(data['emotion']).values

    # Convert pixels to numpy arrays
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(48, 48)
        faces.append(face)

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)  # Add channel dimension

    # Normalize pixel values
    faces = faces.astype('float32')
    faces /= 255.0

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_fer2013()
    print("Data loaded successfully")
    print("Training data shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Testing labels shape:", y_test.shape)