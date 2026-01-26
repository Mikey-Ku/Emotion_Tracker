"""
Emotion Model Training Script
==============================

This script trains a deep learning model for emotion detection using the FER2013 dataset.

The FER2013 dataset contains:
- 35,887 grayscale 48x48 pixel face images
- 7 emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

Training process:
1. Load and preprocess the dataset
2. Create data augmentation pipeline
3. Build CNN model architecture
4. Train with callbacks (early stopping, model checkpointing)
5. Evaluate and save the model
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os


class EmotionModelTrainer:
    """
    Train an emotion detection model.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to FER2013 dataset CSV file
        """
        self.data_path = data_path
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.num_classes = len(self.emotion_labels)
        self.img_size = 48
        
        print("Emotion Model Trainer initialized")
        print(f"Training for {self.num_classes} emotion classes")
    
    def load_fer2013_data(self):
        """
        Load and preprocess the FER2013 dataset.
        
        FER2013 format:
        - CSV with columns: emotion, pixels, Usage
        - pixels: space-separated pixel values (2304 values for 48x48 image)
        - emotion: 0-6 (corresponding to emotion labels)
        - Usage: Training, PublicTest, or PrivateTest
        
        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        if not self.data_path or not os.path.exists(self.data_path):
            print("Error: FER2013 dataset not found!")
            print("\nTo download FER2013:")
            print("1. Visit: https://www.kaggle.com/datasets/msambare/fer2013")
            print("2. Download fer2013.csv")
            print("3. Place it in the project directory")
            return None
        
        print(f"Loading data from {self.data_path}...")
        
        # Read CSV
        df = pd.read_csv(self.data_path)
        
        print(f"Total samples: {len(df)}")
        
        # Parse pixel data
        def parse_pixels(pixel_string):
            """Convert space-separated pixel string to numpy array."""
            return np.array([int(pixel) for pixel in pixel_string.split()])
        
        # Process data
        X = np.array([parse_pixels(pixels) for pixels in df['pixels']])
        X = X.reshape(-1, self.img_size, self.img_size, 1)  # Reshape to (N, 48, 48, 1)
        X = X.astype('float32') / 255.0  # Normalize to [0, 1]
        
        # One-hot encode labels
        y = keras.utils.to_categorical(df['emotion'], self.num_classes)
        
        # Split by usage
        train_mask = df['Usage'] == 'Training'
        val_mask = df['Usage'] == 'PublicTest'
        test_mask = df['Usage'] == 'PrivateTest'
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Print class distribution
        print("\nClass distribution (training set):")
        for i, label in enumerate(self.emotion_labels):
            count = np.sum(df[train_mask]['emotion'] == i)
            percentage = count / len(X_train) * 100
            print(f"  {label:10s}: {count:5d} ({percentage:5.1f}%)")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def create_data_augmentation(self):
        """
        Create data augmentation pipeline.
        
        Data augmentation helps prevent overfitting by creating variations of training images:
        - Rotation: ±20 degrees
        - Width/Height shift: ±10%
        - Zoom: ±10%
        - Horizontal flip: 50% chance
        
        Returns:
            ImageDataGenerator for training
        """
        train_datagen = ImageDataGenerator(
            rotation_range=20,           # Randomly rotate images
            width_shift_range=0.1,       # Randomly shift horizontally
            height_shift_range=0.1,      # Randomly shift vertically
            zoom_range=0.1,              # Randomly zoom
            horizontal_flip=True,        # Randomly flip horizontally
            fill_mode='nearest'          # Fill empty pixels after transformations
        )
        
        # Validation data should not be augmented
        val_datagen = ImageDataGenerator()
        
        return train_datagen, val_datagen
    
    def build_model(self):
        """
        Build the CNN model architecture.
        
        Architecture:
        - 4 Convolutional blocks with increasing filters (64 → 128 → 256 → 512)
        - Each block: Conv2D → BatchNorm → MaxPooling → Dropout
        - 2 Dense layers with dropout
        - Output: Softmax over 7 emotions
        
        Total parameters: ~5 million
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Block 1: Learn basic features (edges, textures)
            layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         input_shape=(self.img_size, self.img_size, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2: Learn facial parts (eyes, mouth, nose)
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3: Learn complex patterns (facial expressions)
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4: Learn high-level features
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and classify
            layers.Flatten(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE")
        print("="*60)
        model.summary()
        print("="*60 + "\n")
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        """
        Train the model.
        
        Training strategy:
        1. Data augmentation for training set
        2. Early stopping to prevent overfitting
        3. Model checkpointing to save best model
        4. Learning rate reduction on plateau
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            
        Returns:
            Trained model and training history
        """
        print("Starting training...")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        # Build model
        model = self.build_model()
        
        # Create data generators
        train_datagen, val_datagen = self.create_data_augmentation()
        
        # Create data generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=batch_size
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=batch_size
        )
        
        # Callbacks
        callbacks = [
            # Save best model based on validation accuracy
            ModelCheckpoint(
                'best_emotion_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Stop training if validation loss doesn't improve for 10 epochs
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate if validation loss plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Training completed!")
        
        return model, history
    
    def evaluate(self, model, X_test, y_test):
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            X_test, y_test: Test data
        """
        print("\nEvaluating on test set...")
        
        # Overall accuracy
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Per-class accuracy
        predictions = model.predict(X_test, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        print("\nPer-class Accuracy:")
        for i, label in enumerate(self.emotion_labels):
            mask = true_classes == i
            if np.sum(mask) > 0:
                class_acc = np.mean(pred_classes[mask] == true_classes[mask])
                print(f"  {label:10s}: {class_acc*100:.2f}%")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_classes, pred_classes)
        
        self._plot_confusion_matrix(cm)
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix."""
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels)
        plt.xlabel('Predicted', fontsize=12, fontweight='bold')
        plt.ylabel('True', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        print("✓ Confusion matrix saved to confusion_matrix.png")
        plt.show()
    
    def plot_training_history(self, history):
        """
        Plot training history.
        
        Args:
            history: Training history from model.fit()
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Model Accuracy', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(history.history['loss'], label='Training')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Loss', fontweight='bold')
        ax2.set_title('Model Loss', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300)
        print("✓ Training history saved to training_history.png")
        plt.show()


def main():
    """Main training pipeline."""
    print("="*60)
    print("EMOTION DETECTION MODEL TRAINING")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = EmotionModelTrainer(data_path='fer2013.csv')
    
    # Load data
    data = trainer.load_fer2013_data()
    
    if data is None:
        print("\nCannot proceed without dataset.")
        print("Please download FER2013 and try again.")
        return
    
    X_train, y_train, X_val, y_val, X_test, y_test = data
    
    # Train model
    model, history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=64
    )
    
    # Plot training history
    trainer.plot_training_history(history)
    
    # Evaluate
    trainer.evaluate(model, X_test, y_test)
    
    # Save final model
    model.save('emotion_model_final.h5')
    print("\n✓ Final model saved to emotion_model_final.h5")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nYou can now use this model with:")
    print("  tracker = EmotionTracker(model_path='best_emotion_model.h5')")


if __name__ == "__main__":
    main()
