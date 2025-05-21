

# 😊 Emotion Detection Flask App

A Flask-based web application that detects human emotions from uploaded images using OpenCV for face detection and a pre-trained ResNet50 model (via transfer learning) for emotion classification. This project aims to demonstrate how deep learning and web development can be combined to build intelligent applications.

---

## 🔥 Features

* 🎭 Detects emotions such as Happy, Sad, Angry, Neutral, etc. from a face in the uploaded image.
* 📷 Uses OpenCV for accurate and real-time face detection.
* 🤖 Employs ResNet50-based deep learning model trained via transfer learning for emotion classification.
* 🌐 Simple and interactive web interface built with Flask and HTML/CSS.

---

## 📁 Project Structure

```
EmotionDetection/
│
├── static/                       # Contains static files like CSS
│
├── templates/                   # HTML templates
│   ├── index.html               # Upload page
│   └── after.html               # Result display page
│
├── app.py                       # Flask application
├── requirements.txt             # Python dependencies
│
├── deploy.prototxt              # Face detection config file
├── res10_300x300_ssd_iter_140000.caffemodel   # Face detection model
├── haarcascade_frontalface_default.xml        # Alternative face detection model (optional)
├── ResNet50_Transfer_Learning.keras           # Trained ResNet50 model
└── README.md                    # This file
```

---

## 🛠️ Installation

### ✅ Prerequisites

* Python 3.8+
* `pip` for installing packages

### 📦 Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Tanmaygangurde20/emotion-detection-flask-app.git
   cd emotion-detection-flask-app
   ```

2. **Create and Activate a Virtual Environment** *(Optional but Recommended)*:

   ```bash
   python -m venv venv
   source venv/bin/activate     # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Required Libraries**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Required Models**:

   Place the following files in the project root directory:

   * `deploy.prototxt` – [Download Link](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
   * `res10_300x300_ssd_iter_140000.caffemodel` – [Download Link](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel)
   * `ResNet50_Transfer_Learning.keras` – Your trained emotion detection model using ResNet50

---

## 🚀 Running the App

Start the Flask development server:

```bash
python app.py
```

Open your browser and visit:
📍 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 👨‍🔬 How It Works

1. User uploads an image.
2. OpenCV detects the face region using pre-trained DNN model.
3. The detected face is passed to the ResNet50 model.
4. Model predicts the emotion label.
5. Result is displayed with the predicted emotion and the face image.

---

## 🧠 Model Training (Optional)

If you'd like to train your own model:

* Use a dataset like [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) or similar.
* Apply data preprocessing (resizing, normalization).
* Use TensorFlow/Keras with ResNet50 base and a custom dense classifier.
* Save the model as `.keras` and place it in the root directory.

---

## 📷 Screenshots

![Result Page](virat_page.png)

---

## 📄 License

This project is for educational purposes. You are free to use, modify, and distribute it with credit.

---


