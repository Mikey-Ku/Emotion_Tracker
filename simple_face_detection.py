
import cv2


def main(): 
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
    
    # Optional: Set camera resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*60)
    print("Face Detection Running!")
    print("="*60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  '+' - Increase detection sensitivity")
    print("  '-' - Decrease detection sensitivity")
    print("="*60 + "\n")
    
    # Detection sensitivity parameter
    # Lower = more sensitive (more false positives)
    # Higher = less sensitive (might miss some faces)
    min_neighbors = 10
    
    # Frame counter for statistics
    frame_count = 0
    total_faces_detected = 0
    
    # ========================================================================
    # STEP 3: Main Detection Loop
    # ========================================================================
    
    while True:
        # Read a frame from the webcam
        # ret = boolean (True if frame was read successfully)
        # frame = the actual image (numpy array)
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        
        frame_count += 1
        
        # --------------------------------------------------------------------
        # STEP 3A: Prepare the image for face detection
        # --------------------------------------------------------------------
        # Haar Cascade works on grayscale images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Histogram equalization improves contrast
        # This helps detect faces in different lighting conditions
        gray = cv2.equalizeHist(gray)
        
        # --------------------------------------------------------------------
        # STEP 3B: Detect faces
        # --------------------------------------------------------------------
        # detectMultiScale finds objects at different scales
        # Returns: array of rectangles [(x, y, width, height), ...]
        
        faces = face_cascade.detectMultiScale(
            gray,                    # Input image (grayscale)
            scaleFactor=1.1,        # How much to reduce image size at each scale
                                    # 1.1 = reduce by 10% each time
            minNeighbors=min_neighbors,  # How many neighbors each candidate needs
                                         # Higher = fewer false positives
            minSize=(30, 30),       # Minimum face size to detect (in pixels)
            flags=cv2.CASCADE_SCALE_IMAGE  # Search for faces at different scales
        )
        
        # Update statistics
        total_faces_detected += len(faces)
        
        # --------------------------------------------------------------------
        # STEP 3C: Draw bounding boxes around detected faces
        # --------------------------------------------------------------------
        
        for (x, y, w, h) in faces:
            # Each face is represented as:
            # x, y = top-left corner coordinates
            # w, h = width and height of the bounding box
            
            # Draw rectangle around face
            cv2.rectangle(
                frame,              # Image to draw on
                (x, y),            # Top-left corner
                (x + w, y + h),    # Bottom-right corner
                (0, 255, 0),       # Color in BGR (green)
                2                  # Thickness in pixels
            )
            
            # Optional: Draw a circle at the center of the face
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(
                frame,
                (center_x, center_y),
                5,                 # Radius
                (0, 0, 255),      # Color in BGR (red)
                -1                # -1 = filled circle
            )
            
            # Optional: Add label showing face dimensions
            label = f"Face: {w}x{h}"
            cv2.putText(
                frame,
                label,
                (x, y - 10),      # Position (slightly above the box)
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,              # Font scale
                (0, 255, 0),      # Color (green)
                2                 # Thickness
            )
        
        # --------------------------------------------------------------------
        # STEP 3D: Add information overlay
        # --------------------------------------------------------------------
        
        # Display number of faces detected
        info_text = f"Faces: {len(faces)} | Sensitivity: {min_neighbors}"
        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),  # White text
            2
        )
        
        # Add background for better readability
        cv2.rectangle(frame, (5, 5), (350, 45), (0, 0, 0), -1)
        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # --------------------------------------------------------------------
        # STEP 3E: Display the result
        # --------------------------------------------------------------------
        
        cv2.imshow('Face Detection - Press Q to Quit', frame)
        
        # --------------------------------------------------------------------
        # STEP 3F: Handle keyboard input
        # --------------------------------------------------------------------
        
        # Wait 1ms for a key press
        # & 0xFF ensures compatibility across different systems
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Quit the program
            print("\nQuitting...")
            break
            
        elif key == ord('s'):
            # Save screenshot
            from datetime import datetime
            filename = f"face_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Screenshot saved: {filename}")
            
        elif key == ord('+') or key == ord('='):
            # Increase sensitivity (detect more faces, more false positives)
            min_neighbors = max(1, min_neighbors - 1)
            print(f"Sensitivity increased (minNeighbors = {min_neighbors})")
            
        elif key == ord('-') or key == ord('_'):
            # Decrease sensitivity (detect fewer faces, fewer false positives)
            min_neighbors = min(10, min_neighbors + 1)
            print(f"Sensitivity decreased (minNeighbors = {min_neighbors})")
    
    # ========================================================================
    # STEP 4: Cleanup
    # ========================================================================
    
    # Release the webcam
    cap.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    
    # Print statistics
    print("\n" + "="*60)
    print("Session Statistics")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print(f"Total faces detected: {total_faces_detected}")
    if frame_count > 0:
        avg_faces = total_faces_detected / frame_count
        print(f"Average faces per frame: {avg_faces:.2f}")
    print("="*60 + "\n")


# ============================================================================
# Additional Helper Functions
# ============================================================================

def detect_faces_in_image(image_path):
    """
    Detect faces in a static image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of face rectangles [(x, y, w, h), ...]
    """
    # Load the face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display result
    cv2.imshow('Face Detection - Press any key to close', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return faces


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run the main face detection program
    main()
    
    # Example: Detect faces in a static image
    # Uncomment the line below and provide an image path
    # detect_faces_in_image("path/to/your/image.jpg")
