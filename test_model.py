import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

def predict_emotion(image_path):
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return None, None
            
        # Load the saved model
        print("Loading existing model...")
        model = load_model('best_model.keras')
        
        # Load and preprocess the image
        print(f"Processing image: {image_path}")
        img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        prediction = model.predict(img_array)
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Display results
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        plt.title(f'Predicted: {predicted_emotion}\nConfidence: {confidence:.2f}%')
        plt.axis('off')
        plt.show()
        
        # Print detailed probabilities
        print("\nEmotion Probabilities:")
        for emotion, prob in zip(emotion_labels, prediction[0]):
            print(f"{emotion}: {prob*100:.2f}%")
        
        return predicted_emotion, confidence
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def main():
    print("\nWelcome to Emotion Recognition Test!")
    print("------------------------------------")
    print("Instructions:")
    print("1. Enter the full path to your image file")
    print("2. Make sure the image contains a clear face")
    print("3. Type 'q' to quit the program")
    print("\nExample path: test_images/happy.jpg")
    
    while True:
        # Get image path from user
        image_path = input("\nEnter the path to your image (or 'q' to quit): ")
        
        if image_path.lower() == 'q':
            print("Goodbye!")
            break
            
        result = predict_emotion(image_path)
        if result[0] is not None:  # Only print results if prediction was successful
            emotion, confidence = result
            print(f"\nPredicted emotion: {emotion}")
            print(f"Confidence: {confidence:.2f}%")
        else:
            print("\nFailed to make prediction. Please try another image.")

if __name__ == "__main__":
    main()