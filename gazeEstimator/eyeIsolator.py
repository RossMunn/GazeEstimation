import cv2
import dlib
import imutils
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

# Define test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

# Define the test_generator
test_dir = 'C:\\Users\\jrmun\\Desktop\\test_left'  # Change this to the correct test data directory
BATCH_SIZE = 32
target_size = (42, 50)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

# Create class labels
class_labels = {v: k for k, v in test_generator.class_indices.items()}

predictor_path = "C:\\Users\\jrmun\\PycharmProjects\\Disso\\extractorModels\\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def preprocess_eye(image, left_eye_points, padding_ratio=0.2):
    left_eye_region = np.array([(point.x, point.y) for point in left_eye_points])
    x, y, w, h = cv2.boundingRect(left_eye_region)

    # Add padding
    padding_x = int(w * padding_ratio)
    padding_y = int(h * padding_ratio)
    x, y = max(0, x - padding_x), max(0, y - padding_y)
    w, h = min(image.shape[1] - x, w + 2 * padding_x), min(image.shape[0] - y, h + 2 * padding_y)

    left_eye = image[y:y + h, x:x + w]
    left_eye = cv2.resize(left_eye, (50, 42))
    left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
    left_eye = left_eye.astype("float32") / 255.0
    left_eye = img_to_array(left_eye)
    left_eye = np.expand_dims(left_eye, axis=0)

    return left_eye

model_path = 'C:\\Users\\jrmun\\PycharmProjects\\Disso\\Models\\best_eye_gaze_model.h5'
model = load_model(model_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 1)
    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_points = landmarks.parts()[36:42]

        left_eye = preprocess_eye(frame, left_eye_points)

        prediction = model.predict(left_eye)
        eye_direction = class_labels[np.argmax(prediction)]

        cv2.putText(frame, eye_direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Eye Gaze Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


