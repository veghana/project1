import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import os

def check_image(image_path):
    """Check if image exists and can be opened"""
    print(f"\nChecking image at: {image_path}")
    
    if not os.path.exists(image_path):
        print("Error: Image file does not exist!")
        return False
        
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image file!")
        return False
        
    print(f"Image shape: {img.shape}")
    return True

def preprocess_image(image_path):
    try:
        print("\nPreprocessing image...")
        
        # Check image
        if not check_image(image_path):
            return None, None
            
        # Load image
        img = cv2.imread(image_path)
        print("Image loaded successfully")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("Converted to grayscale")
        
        # Detect face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print("\nNo face detected! Please ensure:")
            print("1. The image contains a clear, front-facing face")
            print("2. The lighting is good")
            print("3. The face is not too small or too large")
            print("4. The image is not blurry")
            return None, None
            
        print(f"Found {len(faces)} face(s)")
        
        # Get the largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        print(f"Selected face size: {w}x{h}")
        
        # Extract face
        face = gray[y:y+h, x:x+w]
        
        # Resize to model input size
        face = cv2.resize(face, (48, 48))
        print("Resized to 48x48")
        
        # Normalize
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        
        print("Preprocessing completed successfully")
        return face, img[y:y+h, x:x+w]
        
    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}")
        return None, None

def predict_emotion_enhanced(image_path):
    try:
        print("\nStarting emotion prediction...")
        
        # Load model
        print("Loading model...")
        model = load_model('best_model.keras')
        print("Model loaded successfully")
        
        # Preprocess image
        processed_image, original_face = preprocess_image(image_path)
        if processed_image is None:
            return None, None
            
        # Make prediction
        print("\nMaking prediction...")
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        prediction = model.predict(processed_image, verbose=0)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(prediction[0])[-3:][::-1]
        
        # Display results
        plt.figure(figsize=(12, 4))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_face, cv2.COLOR_BGR2RGB))
        plt.title('Input Image')
        plt.axis('off')
        
        # Predictions bar chart
        plt.subplot(1, 2, 2)
        emotions = [emotion_labels[i] for i in top_3_idx]
        confidences = [prediction[0][i] * 100 for i in top_3_idx]
        
        plt.bar(emotions, confidences)
        plt.title('Top 3 Predictions')
        plt.ylabel('Confidence (%)')
        plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print("\nDetailed Predictions:")
        print("-" * 30)
        for emotion, conf in zip(emotions, confidences):
            print(f"{emotion}: {conf:.2f}%")
            
        return emotion_labels[np.argmax(prediction[0])], np.max(prediction) * 100
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        return None, None

def main():
    print("\nEnhanced Emotion Recognition")
    print("=" * 30)
    print("Tips for better predictions:")
    print("1. Ensure good lighting")
    print("2. Face should be clearly visible")
    print("3. Try to make the expression more pronounced")
    print("4. Center your face in the image")
    
    while True:
        try:
            image_path = input("\nEnter image path (or 'q' to quit): ")
            if image_path.lower() == 'q':
                print("Goodbye!")
                break
                
            result = predict_emotion_enhanced(image_path)
            if result is not None and result[0] is not None:
                emotion, confidence = result
                print(f"\nPrimary Prediction: {emotion}")
                print(f"Confidence: {confidence:.2f}%")
            else:
                print("\nCould not make prediction. Please try another image.")
                
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try another image or enter 'q' to quit.")

if __name__ == "__main__":
    main()