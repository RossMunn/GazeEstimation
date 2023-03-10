import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('C:\\Users\\jrmun\\Desktop\\MPIIGaze\\Data\\Extracted\\p14\\day07\\model.h5')

# Load the face and left eye cascades
face_cascade = cv2.CascadeClassifier('C:\\Users\\jrmun\\PycharmProjects\\Disso\\haarcascadeXML\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\jrmun\\PycharmProjects\\Disso\\haarcascadeXML\\haarcascade_mcs_lefteye.xml')

# Define the minimum width threshold for the extracted eye image
min_width = 62

# Define the dimensions for the preprocessed image data
img_size = (224, 224)


# Define the function to preprocess the image data
def preprocess_image(img):
    img = cv2.resize(img, img_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


# Define the function to extract and preprocess the left eye image
def extract_left_eye(img, eyes):
    for (ex, ey, ew, eh) in eyes:
        # Check if the eye is on the left side of the face
        if ex < img.shape[1] / 2:
            # Extract the eye image
            eye_img = img[ey:ey + eh, ex:ex + ew]
            # Check if the eye image width is larger than the threshold
            if eye_img.shape[1] >= min_width:
                # Preprocess the eye image
                preprocessed_img = preprocess_image(eye_img)
                return preprocessed_img
    return None


# Define the function to process each frame of the video stream
def process_frame(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Iterate through the faces and extract the left eye image
    for (x, y, w, h) in faces:
        # Extract the face image
        face_img = frame[y:y + h, x:x + w]
        # Convert the face image to grayscale
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # Detect eyes in the face image
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5)
        # Extract and preprocess the left eye image
        left_eye_img = extract_left_eye(face_img, eyes)
        if left_eye_img is not None:
            # Run the model prediction on the preprocessed left eye image
            prediction = model.predict(left_eye_img)
            # Print the predicted x and y values
            print('Predicted x:', prediction[0][0])
            print('Predicted y:', prediction[1][0])

    # Display the processed frame
    cv2.imshow('frame', frame)


# Initialize the video capture from the default webcam
cap = cv2.VideoCapture(0)

# Loop through each frame of the video stream
while (True):
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Process the frame
    process_frame(frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()