from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

# Load the model once when the app starts
model = load_model('ResNet50_Transfer_Learning.keras')

# Define the emotion labels
emotion_labels = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
index_to_emotion = {v: k for k, v in emotion_labels.items()}

# Load face detection model
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['POST'])
def after():
    if request.method == 'POST':
        # Save uploaded image to the static directory
        img = request.files['file1']
        img_path = 'static/file.jpg'
        img.save(img_path)

        # Load the image
        img1 = cv2.imread(img_path)
        (h, w) = img1.shape[:2]
        
        # Prepare the image for face detection
        blob = cv2.dnn.blobFromImage(cv2.resize(img1, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # Pass the blob through the network and obtain the detections
        face_net.setInput(blob)
        detections = face_net.forward()

        # Initialize variable for the cropped image
        cropped = None

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence > 0.5:
                # Compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure the bounding box falls within the image
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w - 1, endX), min(h - 1, endY)
                
                # Extract the face ROI and crop the face
                face = img1[startY:endY, startX:endX]
                cv2.rectangle(img1, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cropped = face
                break  # Process only the first detected face

        # Save the image with rectangles
        cv2.imwrite('static/after.jpg', img1)

        if cropped is not None:
            cv2.imwrite('static/cropped.jpg', cropped)
        else:
            return "Error: No face detected."

        # Load the cropped image for prediction
        image = cv2.imread('static/cropped.jpg')
        
        if image is None:
            return "Error: Could not load cropped image."

        # Convert image to RGB and resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # Convert the image to a numpy array and add a batch dimension
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array /= 255.0  # Rescale pixel values to [0, 1]

        # Make prediction using the pre-loaded model
        prediction = model.predict(image_array)

        # Get the emotion label with the highest probability
        predicted_class = np.argmax(prediction, axis=1)
        predicted_emotion = index_to_emotion.get(predicted_class[0], "Unknown Emotion")

        # Pass the image path and prediction to the template
        return render_template('after.html', data=predicted_emotion, image_path='static/after.jpg')

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)