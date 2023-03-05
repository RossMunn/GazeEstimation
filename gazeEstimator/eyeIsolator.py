import cv2
import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('my_model')

# Define the labels
labels = ['forward_look', 'left_look', 'right_look']

def process_frame(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use Haar cascades to detect the left eye
    eye_cascade = cv2.CascadeClassifier('C:\\Users\\jrmun\\PycharmProjects\\GazeEstimation\\haarcascadeXML\\haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in eyes:
        if x < frame.shape[1] // 2:  # Only consider the left eye
            # Draw a bounding box around the eye
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract the left eye region
            eye = gray[y:y + h, x:x + w]

            # Resize the eye image to the input shape of the model
            input_shape = model.layers[0].input_shape[1:3]
            eye = cv2.resize(eye, input_shape)

            # Normalize the pixel values to be between 0 and 1
            eye = eye / 255.0

            # Add an extra dimension for the batch size
            eye = np.expand_dims(eye, axis=-1)
            eye = np.expand_dims(eye, axis=0)

            # Make the prediction using the model
            prediction = model.predict(eye)[0]

            # Convert the predicted probabilities to percentages
            probabilities = np.round(prediction * 100)

            # Get the index of the label with the highest probability
            label_index = np.argmax(prediction)

            # Get the label with the highest probability
            label = labels[label_index]

            # Draw the label and probability on the frame
            cv2.putText(frame, f"{label} ({probabilities[label_index]:.0f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame


# Define the function to capture frames from the webcam and process them
def webcam_eye_tracking():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)

        cv2.imshow('Eye Tracking', processed_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


webcam_eye_tracking()


