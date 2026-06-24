# Maize Pest Vision 007-2025

This tool uses OpenCV to detect Fall Armyworm (FAW) damage on maize leaves. It identifies characteristic 'windowing' and holes by filtering leaf pigments and analyzing irregular contours.

## Installation
1. Ensure Python 3.8+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the detection script on a photo of a maize leaf:
```bash
python detect.py --input path/to/leaf_image.jpg --output results.jpg
```

## Parameters
- `--input`: Path to your image file.
- `--output`: Where to save the processed image.
- `--sensitivity`: Float (0.1 to 1.0). Higher values detect larger damaged areas.