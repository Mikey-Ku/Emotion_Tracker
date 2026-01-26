"""
Simple Emotion Tracker Demo
============================

This is a simplified version that works without a trained deep learning model.
It uses basic computer vision techniques to demonstrate the concept.

Perfect for:
- Testing the face detection pipeline
- Understanding the overall system flow
- Quick demos without model training
"""

import cv2
import numpy as np
from collections import deque
from datetime import datetime


class SimpleEmotionDemo:
    """
    Simplified emotion tracker using basic CV techniques.
    """
    
    def __init__(self):
        """Initialize the demo tracker."""
        self.emotion_labels = ['Happy', 'Sad', 'Neutral', 'Surprised']
        
        # Load face and eye detectors
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        
        self.emotion_history = deque(maxlen=10)
        
        print("✓ Simple Emotion Demo initialized!")
        print("  Note: This uses basic CV, not deep learning")
        print("  For better accuracy, use the full emotion_tracker.py with a trained model")
    
    def detect_emotion_simple(self, face_gray, face_color):
        """
        Detect emotion using simple heuristics.
        
        Rules:
        1. If smile detected → Happy
        2. If eyes wide (large eye regions) → Surprised
        3. If face dark (low brightness) → Sad
        4. Otherwise → Neutral
        
        Args:
            face_gray: Grayscale face region
            face_color: Color face region
            
        Returns:
            emotion, confidence
        """
        h, w = face_gray.shape
        
        # Detect smile
        smiles = self.smile_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        # Calculate brightness
        brightness = np.mean(face_gray)
        
        # Simple rules
        if len(smiles) > 0:
            emotion = 'Happy'
            confidence = 0.7
        elif len(eyes) >= 2:
            # Check if eyes are wide (surprised)
            eye_areas = [ew * eh for (ex, ey, ew, eh) in eyes]
            avg_eye_area = np.mean(eye_areas)
            face_area = h * w
            
            if avg_eye_area > face_area * 0.05:  # Eyes are large relative to face
                emotion = 'Surprised'
                confidence = 0.6
            elif brightness < 100:  # Dark face
                emotion = 'Sad'
                confidence = 0.5
            else:
                emotion = 'Neutral'
                confidence = 0.6
        else:
            emotion = 'Neutral'
            confidence = 0.5
        
        return emotion, confidence
    
    def run(self):
        """Run the simple demo."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\n" + "="*50)
        print("Simple Emotion Tracker Demo")
        print("="*50)
        print("This is a basic demo using simple CV techniques")
        print("\nTips for better detection:")
        print("  - Face the camera directly")
        print("  - Ensure good lighting")
        print("  - Make clear facial expressions")
        print("  - SMILE to be detected as Happy!")
        print("\nPress 'q' to quit")
        print("="*50 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract face regions
                face_gray = gray[y:y+h, x:x+w]
                face_color = frame[y:y+h, x:x+w]
                
                # Detect emotion
                emotion, confidence = self.detect_emotion_simple(face_gray, face_color)
                
                # Update history
                self.emotion_history.append(emotion)
                
                # Draw rectangle
                color = self._get_color(emotion)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                # Draw label
                label = f"{emotion}: {confidence:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2
                )
                
                # Draw emotion indicator
                self._draw_emotion_indicator(frame, emotion, x + w + 10, y)
            
            # Draw info panel
            self._draw_info_panel(frame)
            
            # Display
            cv2.imshow('Simple Emotion Demo', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n✓ Demo ended")
        if self.emotion_history:
            from collections import Counter
            counts = Counter(self.emotion_history)
            print("\nDetected emotions:")
            for emotion, count in counts.most_common():
                print(f"  {emotion}: {count}")
    
    def _get_color(self, emotion):
        """Get color for emotion."""
        colors = {
            'Happy': (0, 255, 0),      # Green
            'Sad': (255, 0, 0),        # Blue
            'Neutral': (200, 200, 200), # Gray
            'Surprised': (0, 255, 255)  # Yellow
        }
        return colors.get(emotion, (255, 255, 255))
    
    def _draw_emotion_indicator(self, frame, emotion, x, y):
        """Draw emotion emoji/indicator."""
        emojis = {
            'Happy': '😊',
            'Sad': '😢',
            'Neutral': '😐',
            'Surprised': '😮'
        }
        
        emoji = emojis.get(emotion, '?')
        cv2.putText(
            frame,
            emoji,
            (x, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            self._get_color(emotion),
            3
        )
    
    def _draw_info_panel(self, frame):
        """Draw info panel."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(
            frame,
            "Simple Emotion Demo",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Info
        cv2.putText(
            frame,
            "Basic CV - Not Deep Learning",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1
        )
        
        # Recent emotion
        if self.emotion_history:
            recent = self.emotion_history[-1]
            cv2.putText(
                frame,
                f"Current: {recent}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self._get_color(recent),
                2
            )


if __name__ == "__main__":
    print("="*60)
    print("SIMPLE EMOTION TRACKER DEMO")
    print("="*60)
    print("\nThis is a simplified demo that doesn't require a trained model.")
    print("It uses basic computer vision techniques:")
    print("  - Haar Cascade for face detection")
    print("  - Smile detection for happiness")
    print("  - Eye detection for surprise")
    print("  - Brightness analysis for sadness")
    print("\nFor accurate emotion detection, use emotion_tracker.py")
    print("with a trained deep learning model.")
    print("="*60 + "\n")
    
    demo = SimpleEmotionDemo()
    demo.run()
