import cv2
import tensorflow as tf
import numpy as np

# Load the CNN model
model = tf.keras.models.load_model('C:\\Users\\jrmun\\PycharmProjects\\GazeEstimation\\my_model')

# Create a function to extract the eyes from the image using OpenCV
def extract_eyes(image):
    # Load the pre-trained Haar Cascade classifier for eye detection
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect eyes using the Haar Cascade classifier
    eyes = eye_cascade.detectMultiScale(gray)

    # Extract the regions of interest (ROI) corresponding to the eyes
    for (x,y,w,h) in eyes:
        roi = image[y:y+h, x:x+w]
        yield roi

# Create a function to preprocess the eye images for the CNN model
def preprocess_eye(image):
    # Resize the image to the required input size of the CNN model
    image = cv2.resize(image, (64, 64))

    # Convert the image to a numpy array and normalize the pixel values
    image = np.array(image, dtype=np.float32) / 255.0

    # Add an extra dimension to the array to represent the batch size
    image = np.expand_dims(image, axis=0)

    return image

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Extract the eyes from the frame using the extract_eyes function
    for eye in extract_eyes(frame):
        # Preprocess the eye image for the CNN model
        eye = preprocess_eye(eye)

        # Run the CNN model on the eye image to estimate the eye gaze
        result = model.predict(eye)

        # Get the predicted class label for the eye gaze
        classes = ['close_look', 'left_look', 'right_look', 'forward_look']
        label = classes[np.argmax(result)]

        # Display the label on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Eye Gaze Estimation', frame)

    # Check for key press and break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()