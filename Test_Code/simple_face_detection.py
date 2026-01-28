
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
    
    #Camera Resolution
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
    
    # Detection sensitivity
    min_neighbors = 5
    
    # Frame counter for statistics
    frame_count = 0
    
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
        
        #Face Detection
        faces = face_cascade.detectMultiScale(
            gray,                    # Input image (grayscale)
            scaleFactor=1.1,        #reduce image size by 10%
            minNeighbors=min_neighbors,
            minSize=(30, 30),       # Minimum face size to detect (in pixels)
            flags=cv2.CASCADE_SCALE_IMAGE  # Search for faces at different scales
        )
        
        #draw bounding boxes around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,              
                (x, y),            
                (x + w, y + h),    
                (0, 255, 0),       
                2                 
            )
            
            #face dimensions
            label = f"Face: {w}x{h}"
            cv2.putText(
                frame,
                label,
                (x, y - 10),      
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,             
                (0, 255, 0),      
                2
            )
        
        #Quit message
        cv2.imshow('Face Detection - Press Q to Quit', frame)
        
        #Keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
            
        elif key == ord('s'):
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
    
    # Release the webcam
    cap.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Main Loop
if __name__ == "__main__":
    main()