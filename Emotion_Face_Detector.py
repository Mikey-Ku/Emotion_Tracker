import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

def main(): 

    emotion_labels = [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral"
    ]
        
    print("Loading emotion detection model...")
    emotion_model = load_model("face_model.h5")
    print("✓ Emotion model loaded successfully!")

    # Load the Haar Cascade classifier
    print("Loading face detector...")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Verify the classifier loaded successfully
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier!")
        return
    
    print("✓ Face detector loaded successfully!")
    
    #starting the webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return
    
    print("✓ Webcam opened successfully!")
    
    #Camera Resolution - Higher resolution for better detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*60)
    print("Emotion Detection Running!")
    print("="*60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  '+' - Increase detection sensitivity")
    print("  '-' - Decrease detection sensitivity")
    print("  'r' - Reset emotion smoothing")
    print("="*60 + "\n")
    
    # Detection sensitivity - Lower for better detection
    min_neighbors = 3
    scale_factor = 1.05  # More thorough search
    
    # Frame counter for statistics
    frame_count = 0
    
    # Emotion smoothing - Store recent predictions to reduce jitter
    emotion_history = deque(maxlen=10)  # Keep last 10 predictions
    
    #Main Detection Loop
    while True:

        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        
        frame_count += 1
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Histogram equalization - improves contrast
        gray = cv2.equalizeHist(gray)
        
        #Face Detection - Improved parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,  # More thorough (1.05 instead of 1.1)
            minNeighbors=min_neighbors,  # Lower = more sensitive
            minSize=(48, 48),  # Larger minimum to match model input
            maxSize=(500, 500),  # Prevent unrealistic detections
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        #Process each detected face for emotion recognition
        for (x, y, w, h) in faces:
            # Extract face region from grayscale image
            face_roi = gray[y:y+h, x:x+w]
            
            # Additional preprocessing for better emotion recognition
            # Apply histogram equalization to the face region specifically
            face_roi = cv2.equalizeHist(face_roi)
            
            # Preprocess for FER2013 model (48x48 grayscale)
            # Use INTER_AREA for downsampling (better quality)
            face_resized = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
            
            # Normalize to [0, 1] - matching FER2013 training
            face_normalized = face_resized.astype('float32') / 255.0
            
            # Reshape for model input: (1, 48, 48, 1)
            face_input = face_normalized.reshape(1, 48, 48, 1)
            
            # Predict emotion
            emotion_predictions = emotion_model.predict(face_input, verbose=0)
            emotion_probs = emotion_predictions[0]
            
            # Get top prediction
            emotion_idx = np.argmax(emotion_probs)
            emotion_label = emotion_labels[emotion_idx]
            confidence = emotion_probs[emotion_idx]
            
            # Add to history for smoothing
            emotion_history.append(emotion_label)
            
            # Use smoothed emotion (most common in recent history)
            if len(emotion_history) >= 3:
                # Count occurrences of each emotion in history
                emotion_counts = {}
                for e in emotion_history:
                    emotion_counts[e] = emotion_counts.get(e, 0) + 1
                
                # Get most common emotion
                smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
                
                # Use smoothed emotion if it appears at least 30% of the time
                if emotion_counts[smoothed_emotion] / len(emotion_history) >= 0.3:
                    emotion_label = smoothed_emotion
                    # Recalculate confidence for smoothed emotion
                    confidence = emotion_probs[emotion_labels.index(smoothed_emotion)]
            
            # Choose color based on emotion
            if emotion_label == "Happy":
                color = (0, 255, 0)  # Green
            elif emotion_label in ["Angry", "Disgust"]:
                color = (0, 0, 255)  # Red
            elif emotion_label in ["Sad", "Fear"]:
                color = (255, 0, 0)  # Blue
            elif emotion_label == "Surprise":
                color = (0, 255, 255)  # Yellow
            else:  # Neutral
                color = (200, 200, 200)  # Gray
            
            # Draw bounding box around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Display emotion label with confidence
            label = f"{emotion_label}: {confidence:.2f}"
            
            # Draw background for text
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                color,
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text
                2
            )
            
            # Draw all emotion probabilities as bars (optional - for debugging)
            bar_x = x + w + 10
            bar_y = y
            bar_width = 100
            bar_height = 12
            
            for i, (emotion, prob) in enumerate(zip(emotion_labels, emotion_predictions[0])):
                # Check if bars would go off screen
                if bar_x + bar_width > frame.shape[1] or bar_y + (i+1)*15 > frame.shape[0]:
                    break
                
                # Background bar
                cv2.rectangle(frame, (bar_x, bar_y + i*15), 
                             (bar_x + bar_width, bar_y + i*15 + bar_height), 
                             (50, 50, 50), -1)
                
                # Filled bar based on probability
                filled_width = int(bar_width * prob)
                cv2.rectangle(frame, (bar_x, bar_y + i*15), 
                             (bar_x + filled_width, bar_y + i*15 + bar_height), 
                             color if i == emotion_idx else (100, 100, 100), -1)
                
                # Emotion label
                cv2.putText(frame, f"{emotion[:3]}", 
                           (bar_x + bar_width + 5, bar_y + i*15 + bar_height - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        #Display window
        cv2.imshow('Emotion Detection - Press Q to Quit', frame)
        
        #Keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
            
        elif key == ord('s'):
            from datetime import datetime
            filename = f"emotion_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Screenshot saved: {filename}")
            
        elif key == ord('r'):
            # Reset emotion smoothing
            emotion_history.clear()
            print("✓ Emotion history reset")
            
        elif key == ord('+') or key == ord('='):
            # Increase sensitivity (detect more faces)
            min_neighbors = max(1, min_neighbors - 1)
            scale_factor = max(1.01, scale_factor - 0.02)
            print(f"✓ Sensitivity increased (minNeighbors={min_neighbors}, scaleFactor={scale_factor:.2f})")
            
        elif key == ord('-') or key == ord('_'):
            # Decrease sensitivity (fewer false positives)
            min_neighbors = min(8, min_neighbors + 1)
            scale_factor = min(1.3, scale_factor + 0.02)
            print(f"✓ Sensitivity decreased (minNeighbors={min_neighbors}, scaleFactor={scale_factor:.2f})")
    
    # Release the webcam
    cap.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Main Loop
if __name__ == "__main__":
    main()