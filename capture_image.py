import cv2
mport os
def capture_test_image():
   # Create test_images directory if it doesn't exist
   if not os.path.exists('test_images'):
       os.makedirs('test_images')
   
   # Initialize webcam
   print("Initializing webcam...")
   cap = cv2.VideoCapture(0)
   
   if not cap.isOpened():
       print("Error: Could not open webcam")
       return
   
   print("\nInstructions:")
   print("1. Position your face in the camera")
   print("2. Press SPACE to capture an image")
   print("3. Press Q to quit without capturing")
   
   while True:
       # Read frame from webcam
       ret, frame = cap.read()
       
       if not ret:
           print("Error: Could not read frame")
           break
       
       # Flip the frame horizontally to reverse direction
       frame = cv2.flip(frame, 0)  # Changed to 0 to flip vertically only
       
       # Display the frame
       cv2.imshow('Capture Test Image (Press SPACE to capture, Q to quit)', frame)
       
       # Check for key press
       key = cv2.waitKey(1) & 0xFF
       
       # If 'q' is pressed, quit
       if key == ord('q'):
           print("Quitting without capturing...")
           break
           
       # If space is pressed, save the image
       elif key == ord(' '):
           # Convert to grayscale
           gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           
           # Save the image
           image_path = 'test_images/test_image.jpg'
           cv2.imwrite(image_path, gray)
           print(f"\nImage saved to: {image_path}")
           break
   
   # Release webcam and close windows
   cap.release()
   cv2.destroyAllWindows()
   
   return image_path if key == ord(' ') else None
if __name__ == "__main__":
   image_path = capture_test_image()
   if image_path:
       print("\nNow you can use this image path in the emotion recognition script:")
       print(f"test_images/test_image.jpg")