import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# Load the Haar cascade for detecting the face and eyes
face_cascade = cv2.CascadeClassifier('C:\\Users\\jrmun\\PycharmProjects\\Disso\\haarcascadeXML\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\jrmun\\PycharmProjects\\Disso\\haarcascadeXML\\haarcascade_mcs_lefteye.xml')

# Load the MobileNetV2 model for left eye detection
model = load_model('C:\\Users\\jrmun\\Desktop\\MPIIGaze\\Data\\Extracted\\p14\\day07\\model.h5')

# Open a video capture object to read from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        left_eye = None
        right_eye = None

        # Loop through each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Check if the eye is on the left side of the face
            if ex < w/2:
                # Set left_eye variable if not already set
                if left_eye is None:
                    left_eye = (ex, ey, ew, eh)
                # Otherwise, compare positions to see which eye is more left
                else:
                    if ex < left_eye[0]:
                        left_eye = (ex, ey, ew, eh)
            # Otherwise, the eye is on the right side of the face
            else:
                # Set right_eye variable if not already set
                if right_eye is None:
                    right_eye = (ex, ey, ew, eh)

        # If only the left eye was detected, preprocess the image and use the model to predict the x and y coordinates
        if left_eye is not None and right_eye is None:
            (ex, ey, ew, eh) = left_eye

            # Extract the region of interest (ROI) for the left eye
            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]

            # Convert the grayscale eye image to RGB
            eye_rgb = cv2.cvtColor(eye_gray, cv2.COLOR_GRAY2RGB)

            try:
                # Preprocess the eye image for input to the model
                eye_resized = cv2.resize(eye_rgb, (224, 224))
                eye_array = img_to_array(eye_resized)
                eye_array = np.expand_dims(eye_array, axis=0)
                eye_array = preprocess_input(eye_array)

                # Use the model to predict the x and y coordinates of the eye
                predicted_position = model.predict(eye_array)[0]
                predicted_x = int(predicted_position[0] * ew + ex)
                predicted_y = int(predicted_position[1] * eh + ey)

                # Draw a thicker bounding box around the left eye
                thickness = 2
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), thickness)

                # Add text showing the predicted x and y coordinates
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (255, 255, 255)
                line_type = 2
                cv2.putText(roi_color, f'({predicted_x}, {predicted_y})', (predicted_x, predicted_y), font, font_scale,
                            font_color, line_type)

            except Exception as e:
                print(f'Error predicting eye position: {e}')

        # Display the frame with bounding box around the left eye
        cv2.imshow('frame', frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
