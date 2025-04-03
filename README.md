# Chop That Image

A Python utility to cut images into grids of tiles. Useful for splitting sprite sheets, tileset images, or any image into a grid of smaller images.

## Features

- Cut images into customizable grid layouts (e.g., 4x4, 5x5, or any NxM)
- Preview grid lines before cutting
- Batch processing for multiple images
- Options for handling images that don't divide evenly (defaults to **crop**):
  - Scale: Resize image to make it divide evenly
  - Pad: Add padding to make it divide evenly
  - Crop: Cut off extra pixels or interactively select a crop region
- Automatically trim empty borders from each tile (defaults to **enabled** with threshold 75)
- Clean, forest-themed GUI interface
- Command-line interface for scripting and automation

## Installation

1. Clone or download this repository
2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

### Graphical User Interface (GUI)

To launch the GUI, simply run:
```
python image_cutter.py
```

The GUI provides:
- Image selection with preview
- Grid size configuration (rows and columns)
- Options for handling uneven divisions
- Output directory selection
- Batch processing mode

### Command Line Interface (CLI)

For command-line usage:
```
python image_cutter.py --image <path_to_image> --rows <num_rows> --cols <num_cols> [options]
```

#### Command-line options:

- `--image`: Path to the input image
- `--rows`: Number of rows in the grid (default: 3)
- `--cols`: Number of columns in the grid (default: 3)
- `--output`: Output directory (default: "output")
- `--handle-uneven`: How to handle images that don't divide evenly:
  - "scale": Resize the image
  - "pad": Add padding
  - "crop": Cut off extra pixels (default)
- `--no-trim-borders`: Disable automatically trimming empty space around each tile (Trimming is enabled by default)
- `--trim-threshold`: Color difference threshold for trimming (default: 75)
- `--batch`: Enable batch processing mode
- `--gui`: Launch the GUI interface

#### Examples:

Cut an image into a 4x4 grid:
```
python image_cutter.py --image my_image.png --rows 4 --cols 4
```

Process all images in a directory with 5x5 grid:
```
python image_cutter.py --image my_images_folder/ --rows 5 --cols 5 --batch
```

## License

MIT 