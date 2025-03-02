#!/usr/bin/env python3
"""
Basic Cat Detector

A simple implementation of a cat detector using OpenCV's Haar Cascade classifier.
This script captures video from your webcam, detects cats, and saves photos when cats are detected.
"""

import cv2
import os
import time
import subprocess
import platform
from datetime import datetime

def play_sound(sound_file):
    """Play a sound file using the system's audio player"""
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
    """Save a photo with timestamp"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"cat_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Cat photo saved: {filename}")
        return True
    except Exception as e:
        print(f"Error saving photo: {e}")
        return False

def main():
    # Create output directory
    output_dir = "detected_cats"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Load the cat cascade classifier
    print("Loading cat detection model...")
    cascade_path = os.path.join("resources", "haarcascade_frontalcatface.xml")
    if not os.path.exists(cascade_path):
        # Try OpenCV's built-in path as fallback
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalcatface.xml'
        if not os.path.exists(cascade_path):
            print("Error: Could not find Haar cascade file")
            cap.release()
            return
    
    cat_cascade = cv2.CascadeClassifier(cascade_path)
    if cat_cascade.empty():
        print("Error: Could not load cat detection model")
        cap.release()
        return
    
    # Detection parameters
    scale_factor = 1.1
    min_neighbors = 5
    min_size = (50, 50)
    
    # Cooldown settings
    cooldown_period = 5  # seconds
    last_detection_time = 0
    
    # Sound file path
    sound_file = os.path.join("resources", "pug-woof-2-103762.mp3")
    if not os.path.exists(sound_file):
        print(f"Warning: Sound file not found at {sound_file}")
        sound_file = None
    
    print("\nStarting cat detection. Press 'q' to quit.")
    print("Note: Detection works best when:")
    print("- The cat is facing the camera")
    print("- There is good lighting")
    print("- The cat's face is clearly visible")
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            try:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply histogram equalization to improve contrast
                gray = cv2.equalizeHist(gray)
                
                # Detect cats
                cats = cat_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=min_size
                )
                
                # Process detections
                current_time = time.time()
                for (x, y, w, h) in cats:
                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Cat", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Take action if cooldown period has passed
                    if current_time - last_detection_time > cooldown_period:
                        print("\n🐱 Cat detected!")
                        if save_photo(frame, output_dir):
                            if sound_file:
                                play_sound(sound_file)
                            last_detection_time = current_time
                
                # Display frame
                cv2.imshow('Basic Cat Detector', frame)
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break
            
            # Small delay to reduce CPU usage
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 