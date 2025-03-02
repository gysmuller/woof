# WOOF 🐶🙀

A collection of Python scripts for detecting cats using a webcam, with varying levels of complexity and features.

## Overview

This project provides implementations of a cat detection system that:
- Uses your webcam to detect cats
- Plays a sound when a cat is detected
- Saves photos of detected cats
- Provides visual feedback with bounding boxes around detected cats

## Scripts

The project includes the following implementations:

### Basic Implementation
- `basic_cat_detector.py` - Simple implementation using Haar Cascade classifier

### Advanced Implementations
- `safe_cat_detector.py` - Implementation with additional safety features
- `advanced_cat_detector.py` - Most sophisticated implementation using YOLOv3-tiny for better detection

### Utility Scripts
- `download_model.py` - Downloads required model files
- `download_yolo.py` - Downloads YOLOv3-tiny model files
- `build_windows.py` - Script for building Windows executable

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Download required models:
   ```
   python scripts/utils/download_model.py
   ```
   For advanced detection:
   ```
   python scripts/utils/download_yolo.py
   ```

## Usage

### Using the Main Script

The easiest way to run any version of the cat detector is to use the main.py script:

```
python main.py --version [VERSION]
```

Where `[VERSION]` can be one of:
- `basic` - Basic implementation
- `safe` - Safe implementation
- `advanced` - Advanced implementation (default)

Example:
```
python main.py --version basic
```

Or simply run the advanced version (default):
```
python main.py
```

### Running Scripts Directly

You can also run any script directly:

#### Basic Detection
```
python scripts/basic_cat_detector.py
```

#### Safe Detection
```
python scripts/advanced/safe_cat_detector.py
```

#### Advanced Detection
```
python scripts/advanced/advanced_cat_detector.py
```

### Windows Executable
You can run the batch file:
```
run_cat_detector.bat
```

Or build your own executable:
```
python scripts/utils/build_windows.py
```

## Project Structure

```
cat-detector/
├── scripts/
│   ├── advanced/
│   │   ├── safe_cat_detector.py
│   │   └── advanced_cat_detector.py
│   ├── utils/
│   │   ├── download_model.py
│   │   ├── download_yolo.py
│   │   └── build_windows.py
│   └── basic_cat_detector.py
├── resources/
│   ├── haarcascade_frontalcatface.xml
│   ├── pug-woof-2-103762.mp3
│   └── models/
│       ├── coco.names
│       ├── yolov3-tiny.cfg
│       ├── yolov3-tiny.weights
│       ├── yolov3.cfg
│       └── yolov3.weights
├── detected_cats/
├── main.py
├── requirements.txt
├── run_cat_detector.bat
└── README.md
```

## Requirements

- Python 3.6+
- OpenCV
- NumPy

See `requirements.txt` for complete list.

## License

MIT

## Acknowledgements

- OpenCV for computer vision capabilities
- YOLOv3-tiny model for advanced detection
- Haar Cascade classifiers for basic detection 
