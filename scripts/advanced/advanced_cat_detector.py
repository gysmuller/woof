import cv2
import numpy as np
import os
import time
from datetime import datetime
import subprocess
import platform
import atexit
import gc
import sys

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

class AdvancedCatDetector:
    def __init__(self):
        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        # Register cleanup function
        atexit.register(self.cleanup_resources)
        
        # Set conservative camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Create directory for saving cat photos if it doesn't exist
        self.output_dir = "detected_cats"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Load YOLOv3-tiny model
        self.model_file = get_resource_path(os.path.join("resources", "models", "yolov3-tiny.weights"))
        self.config_file = get_resource_path(os.path.join("resources", "models", "yolov3-tiny.cfg"))
        self.names_file = get_resource_path(os.path.join("resources", "models", "coco.names"))
        
        # Load the neural network
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.config_file):
                self.net = cv2.dnn.readNetFromDarknet(self.config_file, self.model_file)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
                # Get the output layer names
                self.layer_names = self.net.getLayerNames()
                self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
                
                # Load class names
                with open(self.names_file, "r") as f:
                    self.classes = [line.strip() for line in f.readlines()]
                
                # Cat class index in COCO dataset
                self.cat_class_id = self.classes.index("cat")
                print("Successfully loaded YOLOv3-tiny model")
            else:
                raise FileNotFoundError("YOLO model files not found")
            
        except Exception as e:
            print(f"Error loading YOLOv3-tiny model: {e}")
            print("Falling back to Haar cascade classifier.")
            self.net = None
            cascade_path = get_resource_path(os.path.join("resources", "haarcascade_frontalcatface.xml"))
            self.cat_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Path to the sound file
        self.sound_file = get_resource_path(os.path.join("resources", "pug-woof-2-103762.mp3"))
        if not os.path.exists(self.sound_file):
            print(f"Warning: Sound file not found")
            self.sound_file = None
        
        # Cooldown period (in seconds) to avoid continuous alerts
        self.cooldown_period = 5
        self.last_detection_time = 0
        
        # Detection parameters
        self.confidence_threshold = 0.3  # Lower threshold for tiny model
        self.nms_threshold = 0.4  # Non-maximum suppression threshold
        
        # Detection counter
        self.detection_count = 0

    def cleanup_resources(self):
        """Ensure all resources are properly released"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        gc.collect()

    def play_sound(self):
        """Play sound using system audio player"""
        if not self.sound_file:
            return
            
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.Popen(['afplay', self.sound_file], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            elif platform.system() == 'Linux':
                subprocess.Popen(['paplay', self.sound_file], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            elif platform.system() == 'Windows':
                subprocess.Popen(['start', self.sound_file], shell=True, 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Error playing sound: {e}")

    def save_cat_photo(self, frame):
        """Save a photo of the detected cat"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"cat_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"\n🐱 Cat detected! Photo saved: {filename}")
            return True
        except Exception as e:
            print(f"Error saving photo: {e}")
            return False

    def detect_cats_with_haar(self, frame, gray):
        """Detect cats using Haar cascade as fallback"""
        cats = self.cat_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detected = False
        for (x, y, w, h) in cats:
            # Draw rectangle around the cat
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Cat", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            detected = True
            
        return detected

    def detect_cats_with_dnn(self, frame):
        """Detect cats using YOLOv3-tiny model"""
        height, width = frame.shape[:2]
        
        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Pass the blob through the network
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        # Initialize lists for detected objects
        class_ids = []
        confidences = []
        boxes = []
        
        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for cats with high confidence
                if class_id == self.cat_class_id and confidence > self.confidence_threshold:
                    # Get bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        detected = False
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"Cat: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detected = True
        
        return detected

    def detect_cats(self):
        """Main method to detect cats from webcam feed"""
        print("Starting advanced cat detection. Press 'q' to quit.")
        print("\nTips for best detection:")
        print("- Ensure good lighting")
        print("- Cat should be facing the camera")
        print("- Keep the camera steady")
        print("- Cat's face should be clearly visible\n")
        
        try:
            while True:
                # Capture frame-by-frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    time.sleep(0.1)
                    continue
                
                # Convert to grayscale for Haar cascade (if needed)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect cats using appropriate method
                if self.net is not None:
                    detected = self.detect_cats_with_dnn(frame)
                else:
                    detected = self.detect_cats_with_haar(frame, gray)
                
                # Take action if cat is detected
                current_time = time.time()
                if detected and current_time - self.last_detection_time > self.cooldown_period:
                    self.detection_count += 1
                    
                    # Save photo and play sound
                    if self.save_cat_photo(frame):
                        print(f"Total detections: {self.detection_count}")
                        self.play_sound()
                        self.last_detection_time = current_time
                
                # Display the resulting frame
                try:
                    cv2.imshow('Advanced Cat Detector', frame)
                except Exception as e:
                    print(f"Display error: {e}")
                    continue
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Force garbage collection periodically
                if self.detection_count % 10 == 0:
                    gc.collect()
                
        except KeyboardInterrupt:
            print("\nStopping cat detector...")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self.cleanup_resources()

if __name__ == "__main__":
    try:
        detector = AdvancedCatDetector()
        detector.detect_cats()
    except Exception as e:
        print(f"Error: {e}") 