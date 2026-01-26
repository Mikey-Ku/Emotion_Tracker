"""
Emotion Tracker - Real-time Emotion Detection and Analysis
============================================================

This module provides real-time emotion detection using OpenCV for face detection
and a pre-trained deep learning model for emotion classification.

Key Components:
1. Face Detection: Uses OpenCV's Haar Cascade classifier
2. Emotion Classification: Deep learning model trained on FER2013 dataset
3. Emotion Rating: Calculates emotional scores based on detected emotions
4. Visualization: Real-time display with emotion labels and confidence scores
"""

import cv2
import numpy as np
from collections import deque
from datetime import datetime
import json
import os


class EmotionTracker:
    """
    Main class for tracking emotions in real-time from webcam feed.
    
    Attributes:
        emotion_labels: List of emotion categories the model can detect
        emotion_history: Stores recent emotion detections for smoothing
        emotion_scores: Tracks cumulative emotional ratings
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the Emotion Tracker.
        
        Args:
            model_path: Path to pre-trained emotion detection model (optional)
                       If None, will use a simplified version
        """
        # Define the 7 basic emotions our model will detect
        # Based on Paul Ekman's research on universal emotions
        self.emotion_labels = [
            'Angry',      # 0 - Negative emotion
            'Disgust',    # 1 - Negative emotion
            'Fear',       # 2 - Negative emotion
            'Happy',      # 3 - Positive emotion
            'Sad',        # 4 - Negative emotion
            'Surprise',   # 5 - Neutral/Positive emotion
            'Neutral'     # 6 - Neutral emotion
        ]
        
        # Emotion valence scores (-1 to 1, where -1 is very negative, 1 is very positive)
        self.emotion_valence = {
            'Angry': -0.8,
            'Disgust': -0.7,
            'Fear': -0.6,
            'Happy': 0.9,
            'Sad': -0.8,
            'Surprise': 0.3,
            'Neutral': 0.0
        }
        
        # Emotion arousal scores (0 to 1, where 0 is calm, 1 is highly aroused)
        self.emotion_arousal = {
            'Angry': 0.8,
            'Disgust': 0.6,
            'Fear': 0.9,
            'Happy': 0.7,
            'Sad': 0.3,
            'Surprise': 0.8,
            'Neutral': 0.2
        }
        
        # Initialize face detector using Haar Cascade
        # This is a pre-trained classifier that comes with OpenCV
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Store recent emotions for smoothing (reduces jitter)
        # We'll keep the last 10 detections
        self.emotion_history = deque(maxlen=10)
        
        # Track emotion statistics over the session
        self.emotion_counts = {emotion: 0 for emotion in self.emotion_labels}
        self.total_detections = 0
        
        # Overall emotional state tracking
        self.current_valence = 0.0  # Current emotional positivity
        self.current_arousal = 0.0  # Current emotional intensity
        
        # Load or create emotion detection model
        self.model = self._load_model(model_path)
        
        print("✓ Emotion Tracker initialized successfully!")
        print(f"  - Detecting {len(self.emotion_labels)} emotions")
        print(f"  - Face detector loaded")
        print(f"  - Model ready")
    
    def _load_model(self, model_path):
        """
        Load the emotion detection model.
        
        For this implementation, we'll use a pre-trained model.
        If no model exists, we'll download one or create a simple version.
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            Loaded Keras model
        """
        try:
            from tensorflow import keras
            
            if model_path and os.path.exists(model_path):
                # Load existing model
                print(f"Loading model from {model_path}...")
                model = keras.models.load_model(model_path)
                return model
            else:
                # We'll create a simple CNN model structure
                # In practice, you'd train this on FER2013 or similar dataset
                print("Creating new emotion detection model...")
                model = self._create_model()
                return model
                
        except ImportError:
            print("Warning: TensorFlow not available. Using simplified detection.")
            return None
    
    def _create_model(self):
        """
        Create a Convolutional Neural Network for emotion detection.
        
        Architecture:
        - Input: 48x48 grayscale images
        - 4 Convolutional blocks with BatchNorm and Dropout
        - Dense layers for classification
        - Output: 7 emotion probabilities
        
        Returns:
            Compiled Keras model
        """
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            # First Convolutional Block
            # Learns basic features like edges and textures
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            # Learns more complex features like facial parts
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            # Learns high-level features like facial expressions
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten for dense layers
            layers.Flatten(),
            
            # Dense layers for classification
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer: 7 emotions with softmax activation
            # Softmax ensures probabilities sum to 1
            layers.Dense(7, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✓ Model architecture created")
        print(f"  - Total parameters: {model.count_params():,}")
        
        return model
    
    def detect_faces(self, frame):
        """
        Detect faces in the given frame using Haar Cascade classifier.
        
        How it works:
        1. Convert frame to grayscale (Haar Cascade works on grayscale)
        2. Apply histogram equalization to improve contrast
        3. Detect faces using the cascade classifier
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            List of face rectangles [(x, y, w, h), ...]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Improve contrast using histogram equalization
        # This helps detect faces in varying lighting conditions
        gray = cv2.equalizeHist(gray)
        
        # Detect faces
        # Parameters:
        #   scaleFactor: How much the image size is reduced at each scale
        #   minNeighbors: How many neighbors each candidate rectangle should have
        #   minSize: Minimum face size to detect
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces, gray
    
    def predict_emotion(self, face_img):
        """
        Predict emotion from a face image.
        
        Process:
        1. Resize face to 48x48 (model input size)
        2. Normalize pixel values to [0, 1]
        3. Run through neural network
        4. Get probability distribution over emotions
        
        Args:
            face_img: Grayscale face image
            
        Returns:
            emotion_label: Predicted emotion string
            confidence: Confidence score (0-1)
            probabilities: Array of probabilities for all emotions
        """
        if self.model is None:
            # Fallback: Use simple face analysis if model not available
            return self._simple_emotion_detection(face_img)
        
        try:
            # Resize to model input size (48x48)
            face_resized = cv2.resize(face_img, (48, 48))
            
            # Normalize pixel values to [0, 1]
            face_normalized = face_resized / 255.0
            
            # Reshape for model input: (1, 48, 48, 1)
            # 1 = batch size, 48x48 = image size, 1 = grayscale channel
            face_input = face_normalized.reshape(1, 48, 48, 1)
            
            # Get predictions
            predictions = self.model.predict(face_input, verbose=0)
            
            # Get the emotion with highest probability
            emotion_idx = np.argmax(predictions[0])
            emotion_label = self.emotion_labels[emotion_idx]
            confidence = predictions[0][emotion_idx]
            
            return emotion_label, confidence, predictions[0]
            
        except Exception as e:
            print(f"Error in emotion prediction: {e}")
            return "Neutral", 0.5, np.ones(7) / 7
    
    def _simple_emotion_detection(self, face_img):
        """
        Simplified emotion detection using image analysis.
        This is a fallback when the deep learning model isn't available.
        
        Uses basic computer vision techniques:
        - Brightness analysis (smiles are often brighter)
        - Edge detection (different emotions have different edge patterns)
        
        Args:
            face_img: Grayscale face image
            
        Returns:
            emotion_label, confidence, probabilities
        """
        # Calculate basic features
        mean_brightness = np.mean(face_img)
        std_brightness = np.std(face_img)
        
        # Simple heuristic (this is very basic and not accurate)
        # In a real system, you'd use the trained model
        if mean_brightness > 150:
            emotion = "Happy"
        elif mean_brightness < 80:
            emotion = "Sad"
        else:
            emotion = "Neutral"
        
        # Create dummy probabilities
        probs = np.ones(7) * 0.1
        idx = self.emotion_labels.index(emotion)
        probs[idx] = 0.5
        probs = probs / probs.sum()
        
        return emotion, 0.5, probs
    
    def calculate_emotional_rating(self):
        """
        Calculate overall emotional rating based on recent detections.
        
        Returns a composite score considering:
        1. Valence: How positive/negative the emotions are (-1 to 1)
        2. Arousal: How intense/calm the emotions are (0 to 1)
        3. Overall mood score: Weighted combination (0 to 100)
        
        Returns:
            dict with valence, arousal, and mood_score
        """
        if not self.emotion_history:
            return {
                'valence': 0.0,
                'arousal': 0.0,
                'mood_score': 50.0,
                'dominant_emotion': 'Neutral'
            }
        
        # Calculate weighted average of recent emotions
        recent_emotions = list(self.emotion_history)
        
        # Weight more recent emotions higher (exponential decay)
        weights = np.exp(np.linspace(-1, 0, len(recent_emotions)))
        weights = weights / weights.sum()
        
        # Calculate weighted valence and arousal
        valence = sum(
            self.emotion_valence[emotion] * weight
            for emotion, weight in zip(recent_emotions, weights)
        )
        
        arousal = sum(
            self.emotion_arousal[emotion] * weight
            for emotion, weight in zip(recent_emotions, weights)
        )
        
        # Calculate mood score (0-100)
        # Formula: 50 + (valence * 40) + (arousal * 10)
        # This gives more weight to valence (positive/negative)
        mood_score = 50 + (valence * 40) + (arousal * 10)
        mood_score = np.clip(mood_score, 0, 100)
        
        # Find dominant emotion
        from collections import Counter
        emotion_counter = Counter(recent_emotions)
        dominant_emotion = emotion_counter.most_common(1)[0][0]
        
        return {
            'valence': valence,
            'arousal': arousal,
            'mood_score': mood_score,
            'dominant_emotion': dominant_emotion
        }
    
    def draw_results(self, frame, faces, emotions_data):
        """
        Draw detection results on the frame.
        
        Visualizations:
        1. Rectangle around detected face
        2. Emotion label with confidence
        3. Emotion probability bar chart
        4. Overall emotional rating
        
        Args:
            frame: Original BGR frame
            faces: List of face rectangles
            emotions_data: List of (emotion, confidence, probs) for each face
            
        Returns:
            Annotated frame
        """
        # Create a copy to draw on
        output = frame.copy()
        
        # Draw each detected face
        for (x, y, w, h), (emotion, confidence, probs) in zip(faces, emotions_data):
            # Draw rectangle around face
            color = self._get_emotion_color(emotion)
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
            
            # Draw emotion label
            label = f"{emotion}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Background for text
            cv2.rectangle(
                output,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                output,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Draw emotion probability bars on the side
            self._draw_emotion_bars(output, probs, x + w + 10, y)
        
        # Draw overall emotional rating
        rating = self.calculate_emotional_rating()
        self._draw_rating_panel(output, rating)
        
        return output
    
    def _get_emotion_color(self, emotion):
        """
        Get color for emotion visualization.
        
        Color coding:
        - Red tones: Negative emotions
        - Green tones: Positive emotions
        - Blue tones: Neutral emotions
        """
        color_map = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 100, 200),  # Orange-Red
            'Fear': (128, 0, 128),     # Purple
            'Happy': (0, 255, 0),      # Green
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (0, 255, 255), # Yellow
            'Neutral': (200, 200, 200) # Gray
        }
        return color_map.get(emotion, (255, 255, 255))
    
    def _draw_emotion_bars(self, frame, probabilities, x, y):
        """
        Draw probability bars for all emotions.
        
        Args:
            frame: Frame to draw on
            probabilities: Array of 7 probabilities
            x, y: Starting position
        """
        bar_width = 100
        bar_height = 15
        spacing = 5
        
        for i, (emotion, prob) in enumerate(zip(self.emotion_labels, probabilities)):
            # Calculate bar position
            bar_y = y + i * (bar_height + spacing)
            
            # Don't draw if off screen
            if bar_y + bar_height > frame.shape[0]:
                break
            if x + bar_width > frame.shape[1]:
                return
            
            # Draw background
            cv2.rectangle(
                frame,
                (x, bar_y),
                (x + bar_width, bar_y + bar_height),
                (50, 50, 50),
                -1
            )
            
            # Draw filled portion
            filled_width = int(bar_width * prob)
            color = self._get_emotion_color(emotion)
            cv2.rectangle(
                frame,
                (x, bar_y),
                (x + filled_width, bar_y + bar_height),
                color,
                -1
            )
            
            # Draw label
            cv2.putText(
                frame,
                f"{emotion[:3]} {prob:.2f}",
                (x + bar_width + 5, bar_y + bar_height - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
    
    def _draw_rating_panel(self, frame, rating):
        """
        Draw overall emotional rating panel.
        
        Shows:
        - Mood score (0-100)
        - Valence indicator
        - Arousal indicator
        - Dominant emotion
        """
        panel_height = 120
        panel_width = 300
        panel_x = 10
        panel_y = 10
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(
            frame,
            "Emotional Rating",
            (panel_x + 10, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Mood score
        mood_score = rating['mood_score']
        mood_color = self._get_mood_color(mood_score)
        cv2.putText(
            frame,
            f"Mood Score: {mood_score:.1f}/100",
            (panel_x + 10, panel_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            mood_color,
            1
        )
        
        # Valence
        valence_text = f"Valence: {rating['valence']:+.2f} "
        valence_text += "(Positive)" if rating['valence'] > 0 else "(Negative)" if rating['valence'] < 0 else "(Neutral)"
        cv2.putText(
            frame,
            valence_text,
            (panel_x + 10, panel_y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Arousal
        arousal_text = f"Arousal: {rating['arousal']:.2f} "
        arousal_text += "(High)" if rating['arousal'] > 0.6 else "(Medium)" if rating['arousal'] > 0.3 else "(Low)"
        cv2.putText(
            frame,
            arousal_text,
            (panel_x + 10, panel_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Dominant emotion
        cv2.putText(
            frame,
            f"Dominant: {rating['dominant_emotion']}",
            (panel_x + 10, panel_y + 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self._get_emotion_color(rating['dominant_emotion']),
            1
        )
    
    def _get_mood_color(self, mood_score):
        """Get color based on mood score (0-100)."""
        if mood_score >= 70:
            return (0, 255, 0)  # Green (good mood)
        elif mood_score >= 40:
            return (0, 255, 255)  # Yellow (neutral)
        else:
            return (0, 0, 255)  # Red (bad mood)
    
    def save_session_data(self, filename="emotion_session.json"):
        """
        Save session statistics to a JSON file.
        
        Args:
            filename: Output filename
        """
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'total_detections': self.total_detections,
            'emotion_counts': self.emotion_counts,
            'final_rating': self.calculate_emotional_rating()
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"✓ Session data saved to {filename}")
    
    def run(self):
        """
        Main loop: Capture video, detect faces, predict emotions, display results.
        
        Controls:
        - 'q': Quit
        - 's': Save screenshot
        - 'r': Reset statistics
        """
        # Open webcam (0 is usually the default camera)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\n" + "="*50)
        print("Emotion Tracker Running!")
        print("="*50)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'r' - Reset statistics")
        print("="*50 + "\n")
        
        frame_count = 0
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame_count += 1
            
            # Detect faces
            faces, gray = self.detect_faces(frame)
            
            # Process each face
            emotions_data = []
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Predict emotion
                emotion, confidence, probs = self.predict_emotion(face_roi)
                emotions_data.append((emotion, confidence, probs))
                
                # Update tracking
                self.emotion_history.append(emotion)
                self.emotion_counts[emotion] += 1
                self.total_detections += 1
            
            # Draw results
            output = self.draw_results(frame, faces, emotions_data)
            
            # Display
            cv2.imshow('Emotion Tracker', output)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                break
            elif key == ord('s'):
                # Save screenshot
                filename = f"emotion_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, output)
                print(f"✓ Screenshot saved: {filename}")
            elif key == ord('r'):
                # Reset statistics
                self.emotion_history.clear()
                self.emotion_counts = {emotion: 0 for emotion in self.emotion_labels}
                self.total_detections = 0
                print("✓ Statistics reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save session data
        self.save_session_data()
        
        # Print final statistics
        self._print_statistics()
    
    def _print_statistics(self):
        """Print session statistics."""
        print("\n" + "="*50)
        print("Session Statistics")
        print("="*50)
        print(f"Total detections: {self.total_detections}")
        print("\nEmotion Distribution:")
        
        for emotion in self.emotion_labels:
            count = self.emotion_counts[emotion]
            percentage = (count / self.total_detections * 100) if self.total_detections > 0 else 0
            print(f"  {emotion:10s}: {count:4d} ({percentage:5.1f}%)")
        
        rating = self.calculate_emotional_rating()
        print(f"\nFinal Emotional Rating:")
        print(f"  Mood Score: {rating['mood_score']:.1f}/100")
        print(f"  Valence: {rating['valence']:+.2f}")
        print(f"  Arousal: {rating['arousal']:.2f}")
        print(f"  Dominant Emotion: {rating['dominant_emotion']}")
        print("="*50 + "\n")


# Main entry point
if __name__ == "__main__":
    # Create and run the emotion tracker
    tracker = EmotionTracker()
    tracker.run()
