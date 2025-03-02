import os
import urllib.request
import ssl
import sys

def download_yolo_files():
    """Download YOLOv3-tiny model files."""
    print("Downloading YOLOv3-tiny model files...")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join("resources", "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Model files to download
    files = {
        os.path.join(models_dir, "yolov3-tiny.cfg"): "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
        os.path.join(models_dir, "coco.names"): "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }
    
    # Create an SSL context that doesn't verify certificates
    ssl_context = ssl._create_unverified_context()
    
    # Download the smaller files first
    for filename, url in files.items():
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return False
    
    # Download the weights file (much smaller than full YOLOv3)
    weights_file = os.path.join(models_dir, "yolov3-tiny.weights")
    weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
    
    if not os.path.exists(weights_file):
        try:
            print("\nDownloading YOLOv3-tiny weights file...")
            
            def progress_callback(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\rProgress: {percent}%")
                sys.stdout.flush()
            
            urllib.request.urlretrieve(weights_url, weights_file, progress_callback)
            print("\nSuccessfully downloaded weights file")
        except Exception as e:
            print(f"\nError downloading weights file: {e}")
            return False
    else:
        print(f"\n{weights_file} already exists, skipping download")
    
    print("\nAll YOLOv3-tiny files downloaded successfully!")
    return True

if __name__ == "__main__":
    download_yolo_files() 