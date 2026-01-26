"""
Emotion Visualization Module
=============================

This module provides tools to visualize emotion tracking data including:
1. Real-time emotion graphs
2. Emotion distribution charts
3. Emotional timeline analysis
4. Mood score trends
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from datetime import datetime
import pandas as pd


class EmotionVisualizer:
    """
    Visualize emotion tracking data with various charts and graphs.
    """
    
    def __init__(self):
        """Initialize the visualizer with a nice style."""
        # Set style for better-looking plots
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # Emotion colors matching the tracker
        self.emotion_colors = {
            'Angry': '#FF0000',
            'Disgust': '#FF6400',
            'Fear': '#800080',
            'Happy': '#00FF00',
            'Sad': '#0000FF',
            'Surprise': '#FFFF00',
            'Neutral': '#C8C8C8'
        }
    
    def plot_emotion_distribution(self, emotion_counts, save_path=None):
        """
        Create a pie chart showing emotion distribution.
        
        Args:
            emotion_counts: Dictionary of emotion counts
            save_path: Optional path to save the figure
        """
        # Filter out emotions with zero count
        filtered_counts = {k: v for k, v in emotion_counts.items() if v > 0}
        
        if not filtered_counts:
            print("No emotion data to visualize")
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        emotions = list(filtered_counts.keys())
        counts = list(filtered_counts.values())
        colors = [self.emotion_colors[e] for e in emotions]
        
        ax1.pie(counts, labels=emotions, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 12})
        ax1.set_title('Emotion Distribution', fontsize=16, fontweight='bold')
        
        # Bar chart
        ax2.bar(emotions, counts, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Emotion', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title('Emotion Frequency', fontsize=16, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for i, (emotion, count) in enumerate(zip(emotions, counts)):
            ax2.text(i, count + max(counts)*0.02, str(count),
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved emotion distribution to {save_path}")
        
        plt.show()
    
    def plot_valence_arousal_space(self, emotions_history, save_path=None):
        """
        Plot emotions in 2D valence-arousal space.
        
        This is based on the circumplex model of emotions where:
        - X-axis: Valence (negative to positive)
        - Y-axis: Arousal (calm to excited)
        
        Args:
            emotions_history: List of detected emotions over time
            save_path: Optional path to save the figure
        """
        # Define valence and arousal for each emotion
        emotion_coords = {
            'Angry': (-0.8, 0.8),
            'Disgust': (-0.7, 0.6),
            'Fear': (-0.6, 0.9),
            'Happy': (0.9, 0.7),
            'Sad': (-0.8, 0.3),
            'Surprise': (0.3, 0.8),
            'Neutral': (0.0, 0.2)
        }
        
        # Count occurrences
        from collections import Counter
        emotion_counts = Counter(emotions_history)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot each emotion
        for emotion, count in emotion_counts.items():
            if emotion in emotion_coords:
                x, y = emotion_coords[emotion]
                size = count * 100  # Scale bubble size by count
                color = self.emotion_colors[emotion]
                
                ax.scatter(x, y, s=size, c=color, alpha=0.6,
                          edgecolors='black', linewidth=2, label=emotion)
                
                # Add label
                ax.annotate(f'{emotion}\n({count})',
                           xy=(x, y), xytext=(10, 10),
                           textcoords='offset points',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5',
                                   facecolor=color, alpha=0.3))
        
        # Add quadrant labels
        ax.text(0.7, 0.9, 'High Arousal\nPositive', fontsize=12,
               ha='center', style='italic', alpha=0.5)
        ax.text(-0.7, 0.9, 'High Arousal\nNegative', fontsize=12,
               ha='center', style='italic', alpha=0.5)
        ax.text(0.7, -0.9, 'Low Arousal\nPositive', fontsize=12,
               ha='center', style='italic', alpha=0.5)
        ax.text(-0.7, -0.9, 'Low Arousal\nNegative', fontsize=12,
               ha='center', style='italic', alpha=0.5)
        
        # Styling
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Valence (Negative ← → Positive)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Arousal (Calm ← → Excited)', fontsize=14, fontweight='bold')
        ax.set_title('Emotion Circumplex Model', fontsize=16, fontweight='bold')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved valence-arousal plot to {save_path}")
        
        plt.show()
    
    def plot_mood_timeline(self, mood_scores, timestamps=None, save_path=None):
        """
        Plot mood score over time.
        
        Args:
            mood_scores: List of mood scores (0-100)
            timestamps: Optional list of timestamps
            save_path: Optional path to save the figure
        """
        if not mood_scores:
            print("No mood data to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Create x-axis
        if timestamps:
            x = timestamps
            ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        else:
            x = range(len(mood_scores))
            ax.set_xlabel('Sample Number', fontsize=12, fontweight='bold')
        
        # Plot mood score
        ax.plot(x, mood_scores, linewidth=2, color='#2E86AB', label='Mood Score')
        ax.fill_between(x, mood_scores, alpha=0.3, color='#2E86AB')
        
        # Add reference lines
        ax.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Good Mood (70+)')
        ax.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Neutral (40-70)')
        ax.axhline(y=40, color='red', linestyle='--', alpha=0.5)
        
        # Add shaded regions
        ax.fill_between(x, 70, 100, alpha=0.1, color='green')
        ax.fill_between(x, 40, 70, alpha=0.1, color='yellow')
        ax.fill_between(x, 0, 40, alpha=0.1, color='red')
        
        # Styling
        ax.set_ylabel('Mood Score', fontsize=12, fontweight='bold')
        ax.set_title('Mood Score Timeline', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Add statistics
        mean_mood = np.mean(mood_scores)
        std_mood = np.std(mood_scores)
        ax.text(0.02, 0.98, f'Mean: {mean_mood:.1f}\nStd: {std_mood:.1f}',
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved mood timeline to {save_path}")
        
        plt.show()
    
    def create_emotion_report(self, session_file, output_dir='emotion_reports'):
        """
        Create a comprehensive emotion analysis report.
        
        Args:
            session_file: Path to emotion session JSON file
            output_dir: Directory to save report files
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load session data
        with open(session_file, 'r') as f:
            data = json.load(f)
        
        print(f"\nGenerating emotion report from {session_file}...")
        print("="*50)
        
        # Extract data
        emotion_counts = data['emotion_counts']
        final_rating = data['final_rating']
        
        # Generate visualizations
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Emotion distribution
        dist_path = os.path.join(output_dir, f'emotion_distribution_{timestamp}.png')
        self.plot_emotion_distribution(emotion_counts, dist_path)
        
        # 2. Generate summary report
        report_path = os.path.join(output_dir, f'emotion_report_{timestamp}.txt')
        self._generate_text_report(data, report_path)
        
        print("="*50)
        print(f"✓ Report generated successfully!")
        print(f"  - Charts saved to: {output_dir}")
        print(f"  - Text report: {report_path}")
        print("="*50)
    
    def _generate_text_report(self, data, output_path):
        """Generate a text-based emotion report."""
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EMOTION TRACKING SESSION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Session Date: {data['timestamp']}\n")
            f.write(f"Total Detections: {data['total_detections']}\n\n")
            
            f.write("-"*60 + "\n")
            f.write("EMOTION DISTRIBUTION\n")
            f.write("-"*60 + "\n")
            
            total = data['total_detections']
            for emotion, count in sorted(data['emotion_counts'].items(),
                                        key=lambda x: x[1], reverse=True):
                percentage = (count / total * 100) if total > 0 else 0
                bar = "█" * int(percentage / 2)
                f.write(f"{emotion:10s}: {bar:50s} {count:4d} ({percentage:5.1f}%)\n")
            
            f.write("\n" + "-"*60 + "\n")
            f.write("EMOTIONAL RATING\n")
            f.write("-"*60 + "\n")
            
            rating = data['final_rating']
            f.write(f"Mood Score:       {rating['mood_score']:.1f}/100\n")
            f.write(f"Valence:          {rating['valence']:+.2f} ")
            f.write("(Positive)\n" if rating['valence'] > 0 else "(Negative)\n")
            f.write(f"Arousal:          {rating['arousal']:.2f}\n")
            f.write(f"Dominant Emotion: {rating['dominant_emotion']}\n")
            
            f.write("\n" + "-"*60 + "\n")
            f.write("INTERPRETATION\n")
            f.write("-"*60 + "\n")
            
            # Add interpretation
            mood_score = rating['mood_score']
            if mood_score >= 70:
                interpretation = "Overall positive emotional state. High mood score indicates predominantly positive emotions."
            elif mood_score >= 40:
                interpretation = "Neutral emotional state. Balanced mix of emotions."
            else:
                interpretation = "Overall negative emotional state. Consider factors affecting mood."
            
            f.write(f"{interpretation}\n")
            
            f.write("\n" + "="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Create visualizer
    viz = EmotionVisualizer()
    
    # Example: Load and visualize a session file
    import sys
    
    if len(sys.argv) > 1:
        session_file = sys.argv[1]
        viz.create_emotion_report(session_file)
    else:
        print("Usage: python emotion_visualizer.py <session_file.json>")
        print("\nOr use in your own code:")
        print("  viz = EmotionVisualizer()")
        print("  viz.plot_emotion_distribution(emotion_counts)")
