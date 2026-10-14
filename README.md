# Kolam Design Analyzer and Recreator

A Python-based computer vision project that analyzes Kolam patterns and generates SVG representations from input images.

## Features

- Image loading and grayscale conversion using OpenCV
- Dot detection and approximate grid estimation
- Image thresholding and stroke skeletonization
- Graph-based representation of stroke paths
- Rotational symmetry scoring
- Smoothed path generation and SVG export

## Tech Stack

- Python
- OpenCV
- NumPy
- SciPy
- scikit-image
- NetworkX
- SVGWrite

## Installation

Install the required dependencies:

```bash
python -m pip install opencv-python numpy scipy scikit-image svgwrite networkx
```

## Usage

Run the analyzer with an input image and an output SVG filename:

```bash
python kolam1.py k1.jpg output1.svg
```

For the second input image:

```bash
python kolam1.py k2.jpg output2.svg
```

Open the generated SVG file in a web browser to inspect the reconstructed pattern.

## Limitations

The current implementation uses approximate grid detection and path extraction. Results may vary with image quality, perspective distortion, connected strokes, and complex Kolam patterns.

## Future Improvements

- Improve dot detection and grid estimation
- Handle loops and junctions more robustly
- Evaluate reconstruction quality on a larger image dataset
- Improve symmetry-aware reconstruction

