import cv2
import dlib
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.optimizers import SGD
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load face detector and facial landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(
    'C:\\Users\\jrmun\\PycharmProjects\\Disso\\extractorModels\\shape_predictor_68_face_landmarks.dat')

# Load left eye model
model_path_left = 'C:\\Users\\jrmun\\PycharmProjects\\Disso\\Models\\best_eye_gaze_model_left.h5'
model_left = load_model(model_path_left)

# Load right eye model
model_path_right = 'C:\\Users\\jrmun\\PycharmProjects\\Disso\\Models\\best_eye_gaze_model_right.h5'
model_right = load_model(model_path_right)

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

def estimate_gaze(left_eye, right_eye, choice):
    if choice == "1":
        return model_left.predict(left_eye)[0]
    elif choice == "2":
        return model_right.predict(right_eye)[0]
    elif choice == "3":
        left_gaze = model_left.predict(left_eye)[0]
        right_gaze = model_right.predict(right_eye)[0]
        return (left_gaze + right_gaze) / 2.0

def fine_tune_model(model, calibration_data_folder):
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.3)
    train_data = datagen.flow_from_directory(
        calibration_data_folder,
        target_size=(50, 42),
        color_mode="grayscale",
        batch_size=32,
        class_mode="categorical",
        subset='training')

    val_data = datagen.flow_from_directory(
        calibration_data_folder,
        target_size=(50, 42),
        color_mode="grayscale",
        batch_size=32,
        class_mode="categorical",
        subset='validation')

    model.compile(loss="categorical_crossentropy", optimizer=SGD(), metrics=["accuracy"])
    model.fit(train_data, validation_data=val_data, epochs=10)

    return model

# Prompt user for their choice
print("Please choose the eye model to use:")
print("1. Left eye only")
print("2. Right eye only")
print("3. Both eyes with score averaging")
user_choice = input("Your choice (1-3): ")

# Define the paths to your calibration datasets
calibration_data_folder_left = 'C:\\Users\\jrmun\\Desktop\\cali_left'
calibration_data_folder_right = 'C:\\Users\\jrmun\\Desktop\\cali_right'

# Fine-tune the models based on the user's choice
if user_choice == "1":
    print("Fine-tuning left eye model...")
    model_left = fine_tune_model(model_left, calibration_data_folder_left)
elif user_choice == "2":
    print("Fine-tuning right eye model...")
    model_right = fine_tune_model(model_right, calibration_data_folder_right)
elif user_choice == "3":
    print("Fine-tuning both eye models...")
    model_left = fine_tune_model(model_left, calibration_data_folder_left)
    model_right = fine_tune_model(model_right, calibration_data_folder_right)
else:
    print("Invalid choice. Please run the script again.")
    exit(1)

def process_images(images_folder):
    total_images = 0
    correct_predictions = 0
    true_labels = []
    predicted_labels = []
    incorrect_image_paths = []

    # Dictionary to map gaze direction labels to numeric values
    gaze_mapping = {
        'Centre': 0,
        'UpRight': 1,
        'UpLeft': 2,
        'Right': 3,
        'Left': 4,
        'DownRight': 5,
        'DownLeft': 6
    }

    # Iterate over images in the folder
    for subfolder in os.listdir(images_folder):
        subfolder_path = os.path.join(images_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        gaze_direction = subfolder.split(".")[-1]
        if gaze_direction not in gaze_mapping:
            continue

        for image_name in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_name)

            image = cv2.imread(image_path)
            if image is None:
                print(f"Unable to read image: {image_path}")
                continue

            # Detect faces in the image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)

            for face in faces:
                landmarks = landmark_predictor(gray, face)

                # Extract left eye
                left_eye_points = [landmarks.part(i) for i in range(36, 42)]
                left_eye = preprocess_eye(image, left_eye_points)

                # Extract right eye
                right_eye_points = [landmarks.part(i) for i in range(42, 48)]
                right_eye = preprocess_eye(image, right_eye_points)

                # Estimate gaze
                gaze = estimate_gaze(left_eye, right_eye, user_choice)

                # Compare gaze with labels and calculate accuracy
                predicted_gaze_direction = np.argmax(gaze)
                true_label = gaze_mapping[gaze_direction]
                true_labels.append(true_label)
                predicted_labels.append(predicted_gaze_direction)

                if predicted_gaze_direction == true_label:
                    correct_predictions += 1
                else:
                    incorrect_image_paths.append(image_path)

                total_images += 1

    accuracy = correct_predictions / total_images
    print(f"Accuracy: {accuracy:.2%}")

    # Print paths of incorrectly predicted images
    print("Incorrectly predicted images:")
    for image_path in incorrect_image_paths:
        print(image_path)

    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Convert to pandas DataFrame for better visualization
    labels = list(gaze_mapping.keys())
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Display confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Call the function with the path to your evaluation dataset
process_images('C:\\Users\\jrmun\\Desktop\\EvalDataset')

