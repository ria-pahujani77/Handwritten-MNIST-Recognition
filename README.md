# Handwritten Digit Recognition App

## Overview

This project consists of a simple Python application for handwritten digit recognition using a Convolutional Neural Network (CNN). The application allows users to draw digits on a Pygame window, and the CNN model predicts the drawn digit.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [License](#license)

## Requirements

- Python 3.x
- Pygame
- TensorFlow
- NumPy
- OpenCV

## Installation

```bash
git clone https://github.com/yourusername/handwritten-digit-recognition.git
cd handwritten-digit-recognition
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

- Draw digits on the Pygame window.
- Press "n" to clear the window.

## Model Training

The model used in this application is a simple CNN trained on the MNIST dataset. The training code is available in the Jupyter notebook mnist-dataset.ipynb. To train the model:

- Open and run the mnist-dataset.ipynb notebook in the master branch.
- The trained model (handwritten_digit_cnnmodel.h5) will be saved in the master branch.
