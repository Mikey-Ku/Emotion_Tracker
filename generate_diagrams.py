"""
System Architecture Diagram Generator
======================================

This script generates visual diagrams of the emotion tracker architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_architecture_diagram():
    """Create a visual diagram of the system architecture."""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Emotion Tracker System Architecture', 
            ha='center', fontsize=20, fontweight='bold')
    
    # Define colors
    color_input = '#E8F4F8'
    color_cv = '#B8E6F0'
    color_ml = '#88D8E8'
    color_analysis = '#58CAE0'
    color_output = '#28BCD8'
    
    # Layer 1: Input
    input_box = FancyBboxPatch((0.5, 7.5), 1.5, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_input, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 8, 'Webcam\nInput', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # Layer 2: Face Detection
    face_box = FancyBboxPatch((2.5, 7.5), 1.5, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=color_cv,
                              edgecolor='black', linewidth=2)
    ax.add_patch(face_box)
    ax.text(3.25, 8.2, 'Face Detection', ha='center', fontsize=10, fontweight='bold')
    ax.text(3.25, 7.8, 'Haar Cascade', ha='center', fontsize=8, style='italic')
    
    # Layer 3: Preprocessing
    prep_box = FancyBboxPatch((4.5, 7.5), 1.5, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=color_cv,
                              edgecolor='black', linewidth=2)
    ax.add_patch(prep_box)
    ax.text(5.25, 8.2, 'Preprocessing', ha='center', fontsize=10, fontweight='bold')
    ax.text(5.25, 7.8, 'Resize & Normalize', ha='center', fontsize=8, style='italic')
    
    # Layer 4: CNN Model
    cnn_box = FancyBboxPatch((6.5, 7.5), 1.5, 1,
                             boxstyle="round,pad=0.1",
                             facecolor=color_ml,
                             edgecolor='black', linewidth=2)
    ax.add_patch(cnn_box)
    ax.text(7.25, 8.2, 'CNN Model', ha='center', fontsize=10, fontweight='bold')
    ax.text(7.25, 7.8, 'Emotion Classification', ha='center', fontsize=8, style='italic')
    
    # Layer 5: Emotion Output
    emotion_box = FancyBboxPatch((8.5, 7.5), 1, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color_output,
                                 edgecolor='black', linewidth=2)
    ax.add_patch(emotion_box)
    ax.text(9, 8, 'Emotion\nPrediction', ha='center', va='center',
            fontsize=10, fontweight='bold')
    
    # Arrows connecting layers
    arrows = [
        (2, 8, 2.5, 8),
        (4, 8, 4.5, 8),
        (6, 8, 6.5, 8),
        (8, 8, 8.5, 8)
    ]
    
    for x1, y1, x2, y2 in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='black')
        ax.add_patch(arrow)
    
    # Analysis Layer
    ax.text(5, 6.8, 'Emotional Analysis', ha='center', 
            fontsize=14, fontweight='bold')
    
    # Valence box
    valence_box = FancyBboxPatch((1, 5.5), 2, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=color_analysis,
                                edgecolor='black', linewidth=2)
    ax.add_patch(valence_box)
    ax.text(2, 6.2, 'Valence', ha='center', fontsize=10, fontweight='bold')
    ax.text(2, 5.8, 'Positive/Negative', ha='center', fontsize=8, style='italic')
    
    # Arousal box
    arousal_box = FancyBboxPatch((4, 5.5), 2, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=color_analysis,
                                edgecolor='black', linewidth=2)
    ax.add_patch(arousal_box)
    ax.text(5, 6.2, 'Arousal', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 5.8, 'Intensity/Calm', ha='center', fontsize=8, style='italic')
    
    # Mood Score box
    mood_box = FancyBboxPatch((7, 5.5), 2, 1,
                             boxstyle="round,pad=0.1",
                             facecolor=color_analysis,
                             edgecolor='black', linewidth=2)
    ax.add_patch(mood_box)
    ax.text(8, 6.2, 'Mood Score', ha='center', fontsize=10, fontweight='bold')
    ax.text(8, 5.8, '0-100 Rating', ha='center', fontsize=8, style='italic')
    
    # Arrows from emotion to analysis
    analysis_arrows = [
        (9, 7.5, 2, 6.5),
        (9, 7.5, 5, 6.5),
        (9, 7.5, 8, 6.5)
    ]
    
    for x1, y1, x2, y2 in analysis_arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=1.5, color='gray',
                               connectionstyle="arc3,rad=0.3")
        ax.add_patch(arrow)
    
    # Output Layer
    ax.text(5, 4.8, 'Visualization & Output', ha='center',
            fontsize=14, fontweight='bold')
    
    # Display box
    display_box = FancyBboxPatch((1, 3.5), 2.5, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=color_output,
                                edgecolor='black', linewidth=2)
    ax.add_patch(display_box)
    ax.text(2.25, 4.2, 'Real-time Display', ha='center', fontsize=10, fontweight='bold')
    ax.text(2.25, 3.8, 'Video + Annotations', ha='center', fontsize=8, style='italic')
    
    # Charts box
    charts_box = FancyBboxPatch((4, 3.5), 2.5, 1,
                               boxstyle="round,pad=0.1",
                               facecolor=color_output,
                               edgecolor='black', linewidth=2)
    ax.add_patch(charts_box)
    ax.text(5.25, 4.2, 'Charts & Graphs', ha='center', fontsize=10, fontweight='bold')
    ax.text(5.25, 3.8, 'Statistics & Trends', ha='center', fontsize=8, style='italic')
    
    # Data box
    data_box = FancyBboxPatch((7, 3.5), 2, 1,
                             boxstyle="round,pad=0.1",
                             facecolor=color_output,
                             edgecolor='black', linewidth=2)
    ax.add_patch(data_box)
    ax.text(8, 4.2, 'Session Data', ha='center', fontsize=10, fontweight='bold')
    ax.text(8, 3.8, 'JSON Export', ha='center', fontsize=8, style='italic')
    
    # CNN Architecture Detail
    ax.text(5, 2.5, 'CNN Architecture Detail', ha='center',
            fontsize=14, fontweight='bold')
    
    # CNN layers
    cnn_layers = [
        ('Input\n48×48', 0.5),
        ('Conv\n64', 1.5),
        ('Conv\n128', 2.5),
        ('Conv\n256', 3.5),
        ('Conv\n512', 4.5),
        ('Dense\n512', 5.5),
        ('Dense\n256', 6.5),
        ('Output\n7', 7.5)
    ]
    
    for label, x in cnn_layers:
        box = FancyBboxPatch((x, 0.8), 0.8, 1.2,
                            boxstyle="round,pad=0.05",
                            facecolor='#88D8E8',
                            edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 0.4, 1.4, label, ha='center', va='center',
               fontsize=8, fontweight='bold')
    
    # Arrows between CNN layers
    for i in range(len(cnn_layers) - 1):
        x1 = cnn_layers[i][1] + 0.8
        x2 = cnn_layers[i + 1][1]
        arrow = FancyArrowPatch((x1, 1.4), (x2, 1.4),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=1.5, color='black')
        ax.add_patch(arrow)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=color_input, edgecolor='black', label='Input'),
        mpatches.Patch(facecolor=color_cv, edgecolor='black', label='Computer Vision'),
        mpatches.Patch(facecolor=color_ml, edgecolor='black', label='Machine Learning'),
        mpatches.Patch(facecolor=color_analysis, edgecolor='black', label='Analysis'),
        mpatches.Patch(facecolor=color_output, edgecolor='black', label='Output')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ System architecture diagram saved to system_architecture.png")
    plt.show()


def create_emotion_circumplex():
    """Create the emotion circumplex diagram."""
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Define emotion positions
    emotions = {
        'Happy': (0.9, 0.7),
        'Excited': (0.6, 0.9),
        'Surprised': (0.3, 0.8),
        'Angry': (-0.8, 0.8),
        'Fear': (-0.6, 0.9),
        'Disgust': (-0.7, 0.6),
        'Sad': (-0.8, 0.3),
        'Bored': (-0.5, -0.5),
        'Calm': (0.5, -0.5),
        'Content': (0.7, 0.3),
        'Neutral': (0.0, 0.2)
    }
    
    colors = {
        'Happy': '#00FF00',
        'Excited': '#00FF88',
        'Surprised': '#00FFFF',
        'Angry': '#FF0000',
        'Fear': '#800080',
        'Disgust': '#FF6400',
        'Sad': '#0000FF',
        'Bored': '#808080',
        'Calm': '#90EE90',
        'Content': '#98FB98',
        'Neutral': '#C8C8C8'
    }
    
    # Plot emotions
    for emotion, (x, y) in emotions.items():
        ax.scatter(x, y, s=500, c=colors[emotion], alpha=0.6,
                  edgecolors='black', linewidth=2)
        ax.annotate(emotion, xy=(x, y), xytext=(0, 0),
                   textcoords='offset points', ha='center', va='center',
                   fontsize=12, fontweight='bold')
    
    # Add axes
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=2)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=2)
    
    # Labels
    ax.set_xlabel('Valence (Negative ← → Positive)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Arousal (Calm ← → Excited)', fontsize=14, fontweight='bold')
    ax.set_title('Circumplex Model of Emotions', fontsize=16, fontweight='bold')
    
    # Quadrant labels
    ax.text(0.7, 0.9, 'High Arousal\nPositive', fontsize=12,
           ha='center', style='italic', alpha=0.5,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2))
    ax.text(-0.7, 0.9, 'High Arousal\nNegative', fontsize=12,
           ha='center', style='italic', alpha=0.5,
           bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
    ax.text(0.7, -0.9, 'Low Arousal\nPositive', fontsize=12,
           ha='center', style='italic', alpha=0.5,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.2))
    ax.text(-0.7, -0.9, 'Low Arousal\nNegative', fontsize=12,
           ha='center', style='italic', alpha=0.5,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('emotion_circumplex.png', dpi=300, bbox_inches='tight')
    print("✓ Emotion circumplex diagram saved to emotion_circumplex.png")
    plt.show()


if __name__ == "__main__":
    print("Generating system diagrams...")
    print("="*60)
    
    create_architecture_diagram()
    create_emotion_circumplex()
    
    print("="*60)
    print("✓ All diagrams generated successfully!")
