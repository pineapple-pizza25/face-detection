import cv2
import pathlib
import time
import requests
import io
import PIL.Image as Image
import numpy as np
from picamera2 import Picamera2

API_URL = 'https://facial-recognition-api.calmwave-03f9df68.southafricanorth.azurecontainerapps.io/facialrecognition'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

def initialize_camera():
    """Initialize PiCamera with optimal settings for face detection"""
    picam2 = Picamera2()
    
    # Configure camera for preview
    preview_config = picam2.create_preview_configuration(
        main={"size": (240, 180)},  # Lower resolution for better performance
        buffer_count=2
    )
    picam2.configure(preview_config)
    
    picam2.start()
    time.sleep(2)  # Give camera time to warm up
    
    return picam2

def detect_faces(frame, classifier):
    """Detect faces in the frame using the cascade classifier"""
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    return classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

def send_image_to_api(image):
    """Send captured image to the facial recognition API"""
    try:
        # Compress image before sending
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_bytes = io.BytesIO(buffer)
        file = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        
        response = requests.post(API_URL, files=file, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f'API request failed: {e}')
        return None

def main():
    try:
        # Initialize face detection classifier
        clf = cv2.CascadeClassifier(CASCADE_PATH)
        if clf.empty():
            raise RuntimeError("Error loading cascade classifier")

        # Initialize camera
        picam2 = initialize_camera()
        print("Camera initialized successfully")

        face_detected = False
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Convert frame format if necessary (PiCamera might use different format)
            if len(frame.shape) == 2:  # If grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Detect faces
            faces = detect_faces(frame, clf)

            if len(faces) > 0 and not face_detected:
                face_detected = True
                
                # Draw rectangles around faces
                for (x, y, width, height) in faces:
                    cv2.rectangle(frame, (x,y), (x+width, y+width), (255, 255, 0), 2)
                
                # Display frame with detected faces
                cv2.imshow("Detected Face", frame)
                cv2.waitKey(1)
                
                # Wait a moment to ensure good quality capture
                time.sleep(2)
                
                # Capture a new frame for API submission
                api_frame = picam2.capture_array()
                result = send_image_to_api(api_frame)
                if result:
                    print('API Response:', result)
                
                # Reset face detection flag after processing
                face_detected = False
                cv2.destroyWindow("Detected Face")
            else:
                # Show live preview
                cv2.imshow("Preview", frame)

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'picam2' in locals():
            picam2.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()