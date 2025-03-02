import cv2
import os
import time
import datetime
import atexit
import gc
import subprocess
import platform
import numpy as np

def cleanup_resources(cap):
    """Ensure camera resources are properly released"""
    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    gc.collect()

def play_sound(sound_file):
    """Play sound using system audio player"""
    try:
        if platform.system() == 'Darwin':  # macOS
            subprocess.Popen(['afplay', sound_file], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        elif platform.system() == 'Linux':
            subprocess.Popen(['paplay', sound_file], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        elif platform.system() == 'Windows':
            subprocess.Popen(['start', sound_file], shell=True, 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Error playing sound: {e}")

def save_photo(frame, output_dir):
    """Save photo with timestamp"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"cat_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"\n🐱 Cat detected! Photo saved: {filename}")
        return True
    except Exception as e:
        print(f"Error saving photo: {e}")
        return False

def validate_cat_detection(gray, x, y, w, h):
    """Additional validation for cat detection"""
    # Check aspect ratio (cat faces are typically more round than human faces)
    aspect_ratio = w / h
    if not (0.8 <= aspect_ratio <= 1.2):  # Cat faces are usually nearly square
        return False
    
    # Check size relative to frame
    frame_height, frame_width = gray.shape
    relative_size = (w * h) / (frame_width * frame_height)
    if not (0.01 <= relative_size <= 0.25):  # Cat face should be reasonable size
        return False
    
    # Check intensity variance in the detected region
    # Cat faces typically have high contrast between features
    roi = gray[y:y+h, x:x+w]
    variance = np.var(roi)
    if variance < 500:  # Threshold determined empirically
        return False
    
    return True

def main():
    # Initialize variables
    cap = None
    last_detection_time = 0
    detection_cooldown = 3  # seconds
    frame_skip = 2  # Process every nth frame
    frame_count = 0
    detection_count = 0
    
    # Create directory for saving cat photos if it doesn't exist
    output_dir = "detected_cats"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Path to the sound file
    sound_file = os.path.join('..', '..', 'resources', 'pug-woof-2-103762.mp3')
    if not os.path.exists(sound_file):
        print(f"Warning: Sound file not found at {sound_file}")
        sound_file = None
    
    try:
        # Initialize camera with conservative settings
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")
            
        # Register cleanup function
        atexit.register(cleanup_resources, cap)
        
        # Set conservative camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Load the cascade classifiers
        cascade_path = os.path.join('..', '..', 'resources', 'haarcascade_frontalcatface.xml')
        if not os.path.exists(cascade_path):
            # Try OpenCV's built-in path as fallback
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalcatface.xml'
            if not os.path.exists(cascade_path):
                raise FileNotFoundError(f"Cascade file not found: {cascade_path}")
        
        cat_cascade = cv2.CascadeClassifier(cascade_path)
        print("Initialization complete\n")
        
        print("Starting cat detection...")
        print("Tips for best detection:")
        print("- Ensure good lighting")
        print("- Cat should be facing the camera")
        print("- Keep the camera steady")
        print("- Cat's face should be clearly visible\n")
        print("Press 'q' to quit")
        
        while True:
            # Read frame with error checking
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to capture frame")
                time.sleep(0.1)
                continue
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            # Convert to grayscale
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Apply histogram equalization to improve contrast
                gray = cv2.equalizeHist(gray)
            except Exception as e:
                print(f"Error converting frame: {e}")
                continue
            
            # Detect cats with stricter parameters
            try:
                cats = cat_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,      # More precise scaling
                    minNeighbors=8,       # Require more neighbors for stronger detection
                    minSize=(60, 60),     # Minimum size for cat face
                    maxSize=(200, 200)    # Maximum size for cat face
                )
            except Exception as e:
                print(f"Detection error: {e}")
                continue
            
            # Process detections
            current_time = time.time()
            valid_detections = []
            
            for (x, y, w, h) in cats:
                # Perform additional validation
                if validate_cat_detection(gray, x, y, w, h):
                    valid_detections.append((x, y, w, h))
            
            if valid_detections and (current_time - last_detection_time) > detection_cooldown:
                detection_count += 1
                
                # Draw rectangles only for valid detections
                for (x, y, w, h) in valid_detections:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Add confidence indicator
                    cv2.putText(frame, "Cat", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save photo and play sound
                if save_photo(frame, "detected_cats"):
                    print(f"Total detections: {detection_count}")
                    if sound_file:
                        play_sound(sound_file)
                    last_detection_time = current_time
            
            # Display frame
            try:
                cv2.imshow('Cat Detector', frame)
            except Exception as e:
                print(f"Display error: {e}")
                continue
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Force garbage collection periodically
            if frame_count % 30 == 0:
                gc.collect()
            
    except KeyboardInterrupt:
        print("\nStopping cat detector...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        cleanup_resources(cap)

if __name__ == "__main__":
    main() 