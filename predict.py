import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image."""
    try:
        img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array, img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

def predict_emotion(model, image_path):
    """Predict emotion for a single image."""
    # Emotion labels
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Load and preprocess image
    img_array, img = load_and_preprocess_image(image_path)
    if img_array is None:
        return
    
    # Make prediction
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

def main():
    try:
        # Load the saved model
        print("Loading model...")
        model = load_model('emotion_recognition_model.keras')
        print("Model loaded successfully!")

        while True:
            # Get image path from user
            image_path = input("\nEnter the path to your image (or 'q' to quit): ")
            
            if image_path.lower() == 'q':
                break
            
            # Make prediction
            result = predict_emotion(model, image_path)
            
            if result:
                emotion, confidence = result
                print(f"\nPredicted emotion: {emotion}")
                print(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()