
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time
import os
import matplotlib.pyplot as plt
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
def process_image(image_path):
   try:
       print("\nProcessing image...")
       
       # Check if image exists
       if not os.path.exists(image_path):
           print(f"Error: Image not found at {image_path}")
           return
       
       # Load image
       print("Loading image...")
       img = cv2.imread(image_path)
       if img is None:
           print("Error: Could not read image")
           return
       
       # Convert to grayscale
       print("Converting to grayscale...")
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
       # Detect faces
       print("Detecting faces...")
       face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
       faces = face_cascade.detectMultiScale(gray, 1.3, 5)
       
       if len(faces) == 0:
           print("No faces detected in the image!")
           return
       
       # Load model
       print("Loading model...")
       try:
           model = load_model('best_model.keras')
       except:
           print("Error: Could not load model. Make sure 'best_model.keras' exists in the current directory.")
           return
       
       emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
       
       print("Processing detected faces...")
       # Process each face
       for (x, y, w, h) in faces:
           face_roi = gray[y:y+h, x:x+w]
           face_roi = cv2.resize(face_roi, (48, 48))
           face_roi = face_roi.astype('float32') / 255.0
           face_roi = np.expand_dims(face_roi, axis=-1)
           face_roi = np.expand_dims(face_roi, axis=0)
           
           # Make prediction
           prediction = model.predict(face_roi, verbose=0)
           emotion_idx = np.argmax(prediction)
           emotion = emotion_labels[emotion_idx]
           confidence = prediction[0][emotion_idx] * 100
           
           # Draw rectangle and emotion
           cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
           label = f"{emotion}: {confidence:.1f}%"
           cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.9, (0, 255, 0), 2)
           
           # Display results with matplotlib
           plt.figure(figsize=(12, 4))
           plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
           plt.title('Detected Face')
           plt.axis('off')
           plt.show()
           
           # Print detailed results
           print("\nDetailed Predictions:")
           print("-" * 30)
           for i, label in enumerate(emotion_labels):
               print(f"{label}: {prediction[0][i] * 100:.2f}%")
               
   except Exception as e:
       print(f"An error occurred: {str(e)}")
   finally:
       plt.close('all')
       print("\nProcessing complete!")
       
       # Show menu again
       print("\nEmotion Recognition System")
       print("=" * 30)
       print("1. Process Image")
       print("2. Real-time Camera")
       print("3. Exit")
def realtime_emotion_detection():
   cap = None
   try:
       print("Loading model...")
       model = load_model('best_model.keras')
       
       print("Starting webcam...")
       cap = cv2.VideoCapture(0)
       
       if not cap.isOpened():
           print("Error: Could not open webcam")
           return
           
       face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
       emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
       
       print("\nControls:")
       print("- Press 'q' to quit")
       print("- Press 's' to save current frame")
       
       while True:
           ret, frame = cap.read()
           if not ret:
               break
               
           gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           faces = face_cascade.detectMultiScale(gray, 1.3, 5)
           
           for (x, y, w, h) in faces:
               face_roi = gray[y:y+h, x:x+w]
               face_roi = cv2.resize(face_roi, (48, 48))
               face_roi = face_roi.astype('float32') / 255.0
               face_roi = np.expand_dims(face_roi, axis=-1)
               face_roi = np.expand_dims(face_roi, axis=0)
               
               prediction = model.predict(face_roi, verbose=0)
               emotion_idx = np.argmax(prediction)
               emotion = emotion_labels[emotion_idx]
               confidence = prediction[0][emotion_idx] * 100
               
               cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
               label = f"{emotion}: {confidence:.1f}%"
               cv2.putText(frame, label, (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                          (0, 255, 0), 2)
           
           cv2.imshow('Real-time Emotion Recognition', frame)
           
           key = cv2.waitKey(1) & 0xFF
           if key == ord('q'):
               break
           elif key == ord('s'):
               timestamp = time.strftime("%Y%m%d_%H%M%S")
               filename = f"emotion_capture_{timestamp}.jpg"
               cv2.imwrite(filename, frame)
               print(f"\nSaved frame as: {filename}")
   
   except Exception as e:
       print(f"An error occurred: {str(e)}")
   finally:
       if cap is not None:
           cap.release()
       cv2.destroyAllWindows()
       print("\nCamera closed.")
       
       # Show menu again
       print("\nEmotion Recognition System")
       print("=" * 30)
       print("1. Process Image")
       print("2. Real-time Camera")
       print("3. Exit")
def main():
   while True:
       print("\nEmotion Recognition System")
       print("=" * 30)
       print("1. Process Image")
       print("2. Real-time Camera")
       print("3. Exit")
       
       choice = input("\nEnter your choice (1-3): ")
       
       if choice == '1':
           image_path = input("\nEnter the path to your image: ")
           process_image(image_path)
           continue
       
       elif choice == '2':
           realtime_emotion_detection()
           continue
       
       elif choice == '3':
           print("\nAre you sure you want to exit? (y/n)")
           confirm = input().lower()
           if confirm == 'y':
               print("Goodbye!")
               break
           else:
               continue
       else:
           print("Invalid choice! Please try again.")
if __name__ == "__main__":
   try:
       main()
   except KeyboardInterrupt:
       print("\nProgram interrupted by user. Exiting...")
   except Exception as e:
       print(f"\nAn error occurred: {str(e)}")
   finally:
       print("\nGoodbye!")
