import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime

print("="*60)
print("Enhanced Emotion Detection")
print("="*60)
print("Loading emotion detection model...")

# Load the trained model
model_best = load_model('face_model.h5')
print("✓ Emotion model loaded successfully!")

# Classes 7 emotional states
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("✓ Face detector loaded successfully!")

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam!")
    exit()

print("✓ Webcam opened successfully!")

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n" + "="*60)
print("Emotion Detection Running!")
print("="*60)
print("Controls:")
print("  'q' - Quit")
print("  's' - Save screenshot")
print("="*60 + "\n")

# Frame counter
frame_count = 0

def get_emotion_color(emotion):
    """Return color based on emotion."""
    color_map = {
        'Happy': (0, 255, 0),        # Green
        'Angry': (0, 0, 255),        # Red
        'Disgust': (0, 100, 200),    # Orange-Red
        'Fear': (255, 0, 0),         # Blue
        'Sad': (255, 100, 0),        # Light Blue
        'Surprise': (0, 255, 255),   # Yellow
        'Neutral': (200, 200, 200)   # Gray
    }
    return color_map.get(emotion, (255, 255, 255))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame")
        break
    
    frame_count += 1

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y + h, x:x + w]
        
        # Resize the face image to the required input size for the model
        face_image = cv2.resize(face_roi, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = image.img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        face_image = np.vstack([face_image])

        # Predict emotion using the loaded model
        predictions = model_best.predict(face_image, verbose=0)
        emotion_idx = np.argmax(predictions)
        emotion_label = class_names[emotion_idx]
        confidence = predictions[0][emotion_idx]
        
        # Get color based on emotion
        color = get_emotion_color(emotion_label)
        
        # Draw rectangle around the face with emotion-based color
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Create label with emotion and confidence
        label = f"{emotion_label}: {confidence:.2f}"
        
        # Get text size for background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (x, y - label_size[1] - 10),
            (x + label_size[0], y),
            color,
            -1  # Filled
        )
        
        # Draw text in white
        cv2.putText(
            frame, 
            label, 
            (x, y - 5), 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, 
            (255, 255, 255),  # White text
            2
        )
    
    # Add info overlay at top
    info_text = f"Faces: {len(faces)} | Frame: {frame_count}"
    cv2.rectangle(frame, (5, 5), (300, 40), (0, 0, 0), -1)
    cv2.putText(
        frame,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # Display the resulting frame
    cv2.imshow('Enhanced Emotion Detection - Press Q to Quit', frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\nQuitting...")
        break
    elif key == ord('s'):
        filename = f"emotion_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✓ Screenshot saved: {filename}")

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("Session Statistics")
print("="*60)
print(f"Total frames processed: {frame_count}")
print("="*60 + "\n")