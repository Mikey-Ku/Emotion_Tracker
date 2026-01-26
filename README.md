# Emotion Tracker 🎭

Real-time emotion detection system using OpenCV and deep learning to analyze facial expressions from webcam feed.

## Features

- **Real-time face detection** using OpenCV Haar Cascades
- **Emotion classification** into 7 categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Emotional rating system** with valence, arousal, and mood score (0-100)
- **Live visualization** with probability bars and statistics
- **Session tracking** with JSON export
- **Data visualization** tools for analysis

## Installation

```bash
# Clone or download this repository
cd Emotion_Tracker

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Option 1: Simple Demo (No Model Required)

```bash
python simple_demo.py
```

This runs a basic version using computer vision techniques - perfect for testing!

### Option 2: Full Tracker (Requires Trained Model)

**Step 1:** Get a trained model

- **Download** a pre-trained FER2013 model from Kaggle and save as `emotion_model.h5`, OR
- **Train your own:**
  1. Download [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013)
  2. Place `fer2013.csv` in project directory
  3. Run `python train_model.py` (takes 2-4 hours on GPU)

**Step 2:** Run the tracker

```bash
python emotion_tracker.py
```

**Controls:**

- `q` - Quit
- `s` - Save screenshot
- `r` - Reset statistics

## How It Works

```
Webcam → Face Detection → Preprocessing → CNN Model → Emotion Classification → Rating → Display
         (Haar Cascade)   (48x48 resize)  (Deep Learning)  (7 emotions)      (Valence/Arousal)
```

### The 7 Emotions

| Emotion     | Valence | Arousal | Description                      |
| ----------- | ------- | ------- | -------------------------------- |
| 😠 Angry    | -0.8    | 0.8     | Negative, high intensity         |
| 🤢 Disgust  | -0.7    | 0.6     | Negative, medium intensity       |
| 😨 Fear     | -0.6    | 0.9     | Negative, very high intensity    |
| � Happy     | +0.9    | 0.7     | Positive, high intensity         |
| 😢 Sad      | -0.8    | 0.3     | Negative, low intensity          |
| 😮 Surprise | +0.3    | 0.8     | Neutral/positive, high intensity |
| 😐 Neutral  | 0.0     | 0.2     | Neutral, low intensity           |

### Emotional Rating

- **Valence** (-1 to +1): How positive/negative the emotion is
- **Arousal** (0 to 1): How intense/calm the emotion is
- **Mood Score** (0-100): Overall emotional state
  - 70-100: Good mood
  - 40-70: Neutral mood
  - 0-40: Low mood

Formula: `mood_score = 50 + (valence × 40) + (arousal × 10)`

## Usage Examples

### Analyze Session Data

After running the tracker, visualize your session:

```bash
python emotion_visualizer.py emotion_session.json
```

Generates:

- Emotion distribution charts
- Valence-arousal plots
- Comprehensive reports

### Generate System Diagrams

```bash
python generate_diagrams.py
```

Creates visual diagrams of the system architecture and emotion model.

### Use in Your Code

```python
from emotion_tracker import EmotionTracker

# Initialize tracker
tracker = EmotionTracker(model_path='emotion_model.h5')

# Run real-time tracking
tracker.run()

# Or process a single image
import cv2
image = cv2.imread('face.jpg')
faces, gray = tracker.detect_faces(image)

for (x, y, w, h) in faces:
    face_roi = gray[y:y+h, x:x+w]
    emotion, confidence, probs = tracker.predict_emotion(face_roi)
    print(f"Detected: {emotion} ({confidence:.2f})")
```

## Project Structure

```
Emotion_Tracker/
├── simple_demo.py           # Basic demo without ML model
├── emotion_tracker.py       # Main tracker with deep learning
├── train_model.py          # Train your own CNN model
├── emotion_visualizer.py   # Data visualization tools
├── generate_diagrams.py    # Generate architecture diagrams
└── requirements.txt        # Python dependencies
```

## Technical Details

### CNN Architecture

- Input: 48×48 grayscale images
- 4 convolutional blocks (64 → 128 → 256 → 512 filters)
- Batch normalization and dropout for regularization
- Dense layers for classification
- Output: Softmax over 7 emotions

### Training

- Dataset: FER2013 (35,887 images)
- Data augmentation (rotation, shift, zoom, flip)
- Callbacks: Early stopping, model checkpointing, learning rate reduction
- Expected accuracy: 65-70% (matches human agreement on FER2013)

## Customization

### Adjust Emotion Mappings

```python
# In emotion_tracker.py
self.emotion_valence = {
    'Happy': 0.95,  # Make happiness more positive
    'Angry': -0.9,  # Make anger more negative
    # ... customize others
}
```

### Change Detection Sensitivity

```python
# More sensitive (detects more faces)
faces = self.face_cascade.detectMultiScale(
    gray, scaleFactor=1.05, minNeighbors=3
)

# Less sensitive (fewer false positives)
faces = self.face_cascade.detectMultiScale(
    gray, scaleFactor=1.2, minNeighbors=7
)
```

## Troubleshooting

**Camera not opening?**

```python
# Try different camera index
cap = cv2.VideoCapture(1)  # or 2, 3, etc.
```

**Low accuracy?**

- Ensure good lighting
- Face camera directly
- Use trained model (not simple demo)
- Train on more data

**Slow performance?**

- Reduce frame resolution
- Process every Nth frame
- Use GPU acceleration

## Requirements

- Python 3.7+
- OpenCV 4.8+
- TensorFlow 2.15+
- NumPy, Pandas, Matplotlib, Seaborn

See `requirements.txt` for full list.

## License

This project is for educational purposes.

## Acknowledgments

- FER2013 dataset creators
- OpenCV community
- TensorFlow/Keras developers

---

**Built with OpenCV, TensorFlow, and Python**
