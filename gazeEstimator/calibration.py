import cv2
import dlib
import imutils
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import SGD

def get_user_input():
    print("Please choose which model(s) to use for gaze estimation:")
    print("1. Left eye model")
    print("2. Right eye model")
    print("3. Both models (averaging)")
    choice = int(input("Enter the number of your choice: "))
    return choice

user_choice = get_user_input()

# Define test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

# Define the test_generator
test_dir = 'C:\\Users\\jrmun\\Desktop\\test_left'
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

def preprocess_eye(image, eye_points, padding_ratio=0.2):
    eye_region = np.array([(point.x, point.y) for point in eye_points])
    x, y, w, h = cv2.boundingRect(eye_region)

    # Add padding
    padding_x = int(w * padding_ratio)
    padding_y = int(h * padding_ratio)
    x, y = max(0, x - padding_x), max(0, y - padding_y)
    w, h = min(image.shape[1] - x, w + 2 * padding_x), min(image.shape[0] - y, h + 2 * padding_y)

    eye = image[y:y + h, x:x + w]
    eye = cv2.resize(eye, (50, 42))
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye = eye.astype("float32") / 255.0
    eye = img_to_array(eye)
    eye = np.expand_dims(eye, axis=0)

    return eye

def fine_tune_model(model, calibration_dir, batch_size=32, epochs=5, target_size=(42, 50)):
    # Define calibration data generator
    calibration_datagen = ImageDataGenerator(rescale=1./255)

    calibration_generator = calibration_datagen.flow_from_directory(
        calibration_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=True
    )

    # Compile the model with a smaller learning rate for fine-tuning
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(learning_rate=1e-5, momentum=0.9),
                  metrics=['accuracy'])

    # Fine-tune the model
    model.fit(calibration_generator, epochs=epochs)

    return model

# Load left eye model
model_path_left = 'C:\\Users\\jrmun\\PycharmProjects\\Disso\\Models\\best_eye_gaze_model_left.h5'
model_left = load_model(model_path_left)

# Load right eye model
model_path_right = 'C:\\Users\\jrmun\\PycharmProjects\\Disso\\Models\\best_eye_gaze_model_right.h5'
model_right = load_model(model_path_right)

# Fine-tune the models with calibration data
calibration_dir_left = 'C:\\Users\\jrmun\\Desktop\\Calibration_data\\left_eye'
calibration_dir_right = 'C:\\Users\\jrmun\\Desktop\\Calibration_data\\right_eye'

model_left = fine_tune_model(model_left, calibration_dir_left)
model_right = fine_tune_model(model_right, calibration_dir_right)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 1)
    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_points = landmarks.parts()[36:42]
        right_eye_points = landmarks.parts()[42:48]

        left_eye = preprocess_eye(frame, left_eye_points)
        right_eye = preprocess_eye(frame, right_eye_points)

        prediction_left = model_left.predict(left_eye)
        prediction_right = model_right.predict(right_eye)

        if user_choice == 1:
            score = prediction_left
            # Draw a rectangle around the left eye
            left_eye_rect = cv2.boundingRect(np.array([(point.x, point.y) for point in left_eye_points]))
            cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]), (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), (0, 255, 0), 2)
        elif user_choice == 2:
            score = prediction_right
            # Draw a rectangle around the right eye
            right_eye_rect = cv2.boundingRect(np.array([(point.x, point.y) for point in right_eye_points]))
            cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]), (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), (0, 255, 0), 2)
        elif user_choice == 3:
            score = (prediction_left + prediction_right) / 2
            # Draw rectangles around both eyes
            left_eye_rect = cv2.boundingRect(np.array([(point.x, point.y) for point in left_eye_points]))
            cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]), (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), (0, 255, 0), 2)
            right_eye_rect = cv2.boundingRect(np.array([(point.x, point.y) for point in right_eye_points]))
            cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]), (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), (0, 255, 0), 2)
        else:
            print("Invalid choice. Exiting.")
            break

        # Calculate the average score
        upleft_index = list(class_labels.keys())[list(class_labels.values()).index('02.UpLeft')]
        upright_index = list(class_labels.keys())[list(class_labels.values()).index('01.UpRight')]
        downleft_index = list(class_labels.keys())[list(class_labels.values()).index('06.DownLeft')]
        downright_index = list(class_labels.keys())[list(class_labels.values()).index('05.DownRight')]
        left_index = list(class_labels.keys())[list(class_labels.values()).index('04.Left')]
        right_index = list(class_labels.keys())[list(class_labels.values()).index('03.Right')]
        center_index = list(class_labels.keys())[list(class_labels.values()).index('00.Centre')]

        if np.argmax(score) in (upleft_index, upright_index):
            eye_direction = 'Up'
        elif np.argmax(score) in (downleft_index, downright_index):
            eye_direction = 'Down'
        elif np.argmax(score) == left_index:
            eye_direction = 'Left'
        elif np.argmax(score) == right_index:
            eye_direction = 'Right'
        elif np.argmax(score) == center_index:
            eye_direction = 'Centre'
        else:
            # Find the class with the maximum probability
            eye_direction = class_labels[np.argmax(score)]

        cv2.putText(frame, eye_direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Eye Gaze Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


