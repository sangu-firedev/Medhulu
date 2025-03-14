<img src="docs/Medhulu.png" alt="MRI Slice" width="250" height="110">

# Medhulu : MRI Preprocessing and Analysis Toolkit

## Overview
This repository provides a comprehensive toolkit for processing, analyzing, and visualizing MRI data. It includes utilities for file loading, preprocessing, skull stripping, registration, normalization, segmentation, and plotting of MRI scans. 

## Features
- **Load MRI Files**: Supports `.nii.gz`, `.mgz`, `.img`, and `.img.gz` formats.
- **Preprocessing Pipeline**: Includes Z-score and Min-Max normalization, N4 bias field correction, and segmentation of CSF, GM, and WM.
- **Skull Stripping**: Uses SynthStrip via Docker for automated skull removal.
- **Image Registration**: Co-registers images to a common reference.
- **Visualization**: Supports 2D slice plotting with overlays.

## Installation
To use this toolkit, install the required dependencies:
```bash
pip install nibabel numpy matplotlib ants joblib tqdm monai
```
Additionally, install Docker for skull stripping.

## Usage
### 1. Load MRI Files
```python
from utilities.load_files import load_files
images = load_files("path/to/mri/folder")
```

### 2. Load a Single Image Using `LoadImage`
```python
from src.main import LoadImage
image = LoadImage("path/to/mri.nii.gz")
```

### 3. Preprocess MRI Images
```python
from utilities.preprocess import preproces_pipeline
preproces_pipeline("input_folder", "output_folder", workers=4, batch_size=10)
```

### 4. Register Images to a Common Reference
```python
from utilities.preprocess import register_images
register_images("input_folder", "output_folder", "path/to/fixed_image.nii.gz", transform="SyN", thread=4)
```

### 5. Skull Stripping
```python
from utilities.skull_stripping import _skull_strip_files
_skull_strip_files("input_folder", "output_folder", use_gpu=True, threads=4)
```

### 6. Plot MRI Slices
```python
from utilities.plotting import plot_files
plot_files("path/to/mri/folder")
```

## LoadImage Class
The `LoadImage` class provides an object-oriented approach to handling MRI images. It supports loading, visualization, and skull stripping.

### Key Methods:
- `get_img()`: Returns the image in NiBabel format.
- `get_data()`: Retrieves the MRI data as a NumPy array.
- `plot(slice=None, save_path=None)`: Plots the MRI slices.
- `skull_strip(output_dir, load_nib=True)`: Performs skull stripping and optionally reloads the processed image.
- `__call__()`: The `LoadImage` object is callable and returns the MRI data when invoked.

Example:
```python
image = LoadImage("path/to/mri.nii.gz")
```

## File Structure
- `main.py`: Defines the `LoadImage` class for handling MRI images.
- `load_files.py`: Handles file loading and metadata extraction.
- `preprocess.py`: Implements normalization, bias field correction, segmentation, resampling, and image registration.
- `skull_stripping.py`: Uses Docker-based SynthStrip for skull removal.
- `plotting.py`: Generates visualizations for MRI slices and segmented regions.

## License
This project is licensed under the MIT License.

## Acknowledgments
- **ANTsPy** for image processing
- **SynthStrip** for skull stripping
- **NiBabel** for MRI file handling

