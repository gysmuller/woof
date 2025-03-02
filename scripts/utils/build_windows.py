import os
import shutil
import subprocess
import sys

def ensure_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def copy_resource_file(src, dest_dir):
    """Copy a file to the resources directory"""
    if os.path.exists(src):
        dest = os.path.join(dest_dir, os.path.basename(src))
        shutil.copy2(src, dest)
        print(f"Copied: {src} -> {dest}")
        return True
    else:
        print(f"Warning: Source file not found: {src}")
        return False

def create_executable():
    """Create Windows executable using PyInstaller"""
    print("Starting build process...")
    
    # Ensure virtual environment is active
    if 'VIRTUAL_ENV' not in os.environ:
        print("Error: Virtual environment not activated!")
        print("Please activate your virtual environment first:")
        print("  Windows: .\\venv\\Scripts\\activate")
        print("  Unix/MacOS: source venv/bin/activate")
        return False
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print(f"Using PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("Error: PyInstaller not found!")
        print("Please install it using: pip install pyinstaller")
        return False
    
    # Create required directories
    ensure_directory("dist")
    ensure_directory("build")
    resource_dir = os.path.join("dist", "resources")
    ensure_directory(resource_dir)
    models_dir = os.path.join(resource_dir, "models")
    ensure_directory(models_dir)
    
    # Define required files
    required_files = {
        "resources/haarcascade_frontalcatface.xml": True,  # True means required
        "resources/pug-woof-2-103762.mp3": False,  # False means optional
    }
    
    optional_files = {
        "resources/models/yolov3-tiny.cfg": "YOLO model configuration",
        "resources/models/yolov3-tiny.weights": "YOLO model weights",
        "resources/models/coco.names": "YOLO class names"
    }
    
    # Copy required files
    print("\nCopying resource files...")
    missing_required = False
    
    for file, required in required_files.items():
        dest_dir = resource_dir
        if not copy_resource_file(file, dest_dir) and required:
            print(f"Error: Required file missing: {file}")
            missing_required = True
    
    if missing_required:
        print("\nError: Some required files are missing. Cannot continue.")
        return False
    
    # Copy optional files
    print("\nCopying optional files...")
    for file, description in optional_files.items():
        dest_dir = os.path.dirname(os.path.join(resource_dir, file.replace("resources/", "")))
        ensure_directory(dest_dir)
        if not copy_resource_file(file, dest_dir):
            print(f"Note: Optional file missing: {file} ({description})")
            print("The program will fall back to Haar cascade detection.")
    
    # Create icon file if not exists
    icon_path = "cat.ico"
    if not os.path.exists(icon_path):
        print("\nNote: No icon file found. Using default window icon.")
        icon_path = None
    
    # Build PyInstaller command
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",
    ]
    
    if icon_path:
        cmd.extend(["--icon", icon_path])
    
    cmd.extend([
        "--add-data", f"dist/resources{os.pathsep}resources",
        "--name", "CatDetector",
        "scripts/advanced/advanced_cat_detector.py"
    ])
    
    # Run PyInstaller
    print("\nBuilding executable...")
    try:
        subprocess.run(cmd, check=True)
        
        # Copy batch file
        shutil.copy2("run_cat_detector.bat", "dist/run_cat_detector.bat")
        
        print("\nBuild successful!")
        print("\nExecutable and resources created in 'dist' folder:")
        print("- CatDetector.exe (main program)")
        print("- resources/ (required files)")
        print("- run_cat_detector.bat (easy launch script)")
        print("\nTo deploy:")
        print("1. Copy the entire 'dist' folder to the target machine")
        print("2. Run 'run_cat_detector.bat' or 'CatDetector.exe' directly")
        print("\nNote: The program requires a webcam to function.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during build: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

if __name__ == "__main__":
    success = create_executable()
    sys.exit(0 if success else 1) 