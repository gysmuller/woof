import os
import urllib.request
import ssl

def download_model_files():
    """Download the MobileNet SSD model files."""
    print("Downloading MobileNet SSD model files...")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join("resources", "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Model files URLs
    files = {
        os.path.join(models_dir, "mobilenet_ssd.prototxt"): "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt",
        os.path.join(models_dir, "mobilenet_ssd.caffemodel"): "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"
    }
    
    # Create an SSL context that doesn't verify certificates
    ssl_context = ssl._create_unverified_context()
    
    for filename, url in files.items():
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            print("\nPlease download the model files manually:")
            print("1. Go to: https://github.com/chuanqi305/MobileNet-SSD")
            print("2. Download the following files:")
            print("   - MobileNetSSD_deploy.prototxt")
            print("   - MobileNetSSD_deploy.caffemodel")
            print(f"3. Place them in the '{models_dir}' directory as:")
            print(f"   - {os.path.join(models_dir, 'mobilenet_ssd.prototxt')}")
            print(f"   - {os.path.join(models_dir, 'mobilenet_ssd.caffemodel')}")
            return False
    
    print("\nAll model files downloaded successfully!")
    return True

if __name__ == "__main__":
    download_model_files() 