# Emotion Detection Flask App

This is a Flask-based web application that detects human emotions from images using a pre-trained ResNet50 model for transfer learning.


## Features

- Upload an image and detect the emotion of the person in the image.
- Uses OpenCV for face detection.
- Utilizes a pre-trained ResNet50 model for emotion classification.

## Requirements

- Python 3.8+
- Flask
- OpenCV
- NumPy
- TensorFlow

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/emotion-detection-flask-app.git
    cd emotion-detection-flask-app
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

4. Download the required models and place them in the root directory:
    - [ResNet50 Transfer Learning Model](link-to-model)
    - [OpenCV Face Detection Model](link-to-model)
    - Ensure you have `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` in the root directory.

## Running the App

1. Start the Flask server:
    ```sh
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000/`.

## Usage
Create 2 folder in your project direcrory static and  templates

1. Upload an image using the provided form.
2. The app will detect the face, predict the emotion, and display the results.
EmotionDetection/
│
├── static/
│
├── templates/
│   ├── after.html
│   ├── index.html
│
├── app.py
├── deploy.prototxt  # Download from Google
├── haarcascade_frontalface_default.xml  # Download from Google
├── res10_300x300_ssd_iter_140000.caffemodel  # Download from Google
└── ResNet50_Transfer_Learning.keras  # Train the model and save

