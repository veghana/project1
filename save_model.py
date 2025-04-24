import tensorflow as tf
from tensorflow.keras.models import load_model
from data_preparation import load_fer2013
from model import create_model

def save_trained_model():
    try:
        # Load data
        print("Loading data...")
        X_train, X_test, y_train, y_test = load_fer2013()
        
        # Create and compile model
        print("Creating model...")
        model = create_model()
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Train model
        print("Training model...")
        history = model.fit(X_train, y_train,
                          validation_data=(X_test, y_test),
                          epochs=50,
                          batch_size=64)
        
        # Save model
        print("Saving model...")
        model.save('emotion_recognition_model.keras')
        print("Model saved successfully!")
        
        # Evaluate model
        print("\nEvaluating model...")
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_acc:.4f}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    save_trained_model()