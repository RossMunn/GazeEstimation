import cv2
import numpy as np

# Create a VideoCapture object to capture video from the default camera
cap = cv2.VideoCapture(0)

# Load the face and eye cascades for detecting faces and eyes
face_cascade = cv2.CascadeClassifier('C:\\Users\\jrmun\\PycharmProjects\\GazeEstimation\\haarcascadeXML\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\jrmun\\PycharmProjects\\GazeEstimation\\haarcascadeXML\\haarcascade_eye.xml')

# Define the desired size for the eye images
EYE_SIZE = (250, 250)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # For each detected face, detect eyes and display both eyes in one frame
    for (x, y, w, h) in faces:
        # Extract the region of interest corresponding to the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes in the region of interest
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5)

        # Extract both eyes and concatenate them into one image
        left_eye = None
        right_eye = None
        for (ex, ey, ew, eh) in eyes:
            eye = roi_color[ey:ey + eh, ex:ex + ew]
            eye = cv2.resize(eye, EYE_SIZE)
            if ex < w / 2:
                if left_eye is None:
                    left_eye = eye
                else:
                    left_eye = np.hstack((left_eye, eye))
            else:
                if right_eye is None:
                    right_eye = eye
                else:
                    right_eye = np.hstack((right_eye, eye))

        if left_eye is not None and right_eye is not None:
            # Combine both eyes into one image
            eye_frame = np.hstack((left_eye, right_eye))
            cv2.imshow('Eyes', eye_frame)

    # Wait for key press and exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and destroy all windows
cap.release()
cv2.destroyAllWindows()





