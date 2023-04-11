import os
import cv2
import dlib
import numpy as np

# Load models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("C:\\Users\\jrmun\\PycharmProjects\\Disso\\extractorModels\\shape_predictor_68_face_landmarks.dat")


def extract_eye_patches(image, padding_ratio=0.2):
    # Detect faces
    faces = face_detector(image, 1)

    left_eye_patches = []
    right_eye_patches = []

    for face in faces:
        # Detect landmarks
        landmarks = landmark_predictor(image, face)

        # Extract left eye patch
        left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        left_eye_roi = cv2.boundingRect(left_eye_points)

        # Extract right eye patch
        right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        right_eye_roi = cv2.boundingRect(right_eye_points)

        # Add padding
        padding_x = int(left_eye_roi[2] * padding_ratio)
        padding_y = int(left_eye_roi[3] * padding_ratio)
        x, y, w, h = left_eye_roi
        x, y = max(0, x-padding_x), max(0, y - padding_y)
        w, h = min(image.shape[1] - x, w + 2 * padding_x), min(image.shape[0] - y, h + 2 * padding_y)
        left_eye_patch = image[y:y + h, x:x + w]

        padding_x = int(right_eye_roi[2] * padding_ratio)
        padding_y = int(right_eye_roi[3] * padding_ratio)
        x, y, w, h = right_eye_roi
        x, y = max(0, x - padding_x), max(0, y - padding_y)
        w, h = min(image.shape[1] - x, w + 2 * padding_x), min(image.shape[0] - y, h + 2 * padding_y)

        right_eye_patch = image[y:y + h, x:x + w]

        left_eye_patches.append(left_eye_patch)
        right_eye_patches.append(right_eye_patch)

    return left_eye_patches, right_eye_patches

def process_images(input_folder, left_output_folder, right_output_folder, current_folder=""):
    current_input_folder = os.path.join(input_folder, current_folder)
    current_left_output_folder = os.path.join(left_output_folder, current_folder)
    current_right_output_folder = os.path.join(right_output_folder, current_folder)
    for entry in os.scandir(current_input_folder):
        if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = entry.path
            os.makedirs(current_left_output_folder, exist_ok=True)
            os.makedirs(current_right_output_folder, exist_ok=True)

            # Load image
            image = cv2.imread(input_path)
            left_eye_patches, right_eye_patches = extract_eye_patches(image)

            for idx, (left_eye_patch, right_eye_patch) in enumerate(zip(left_eye_patches, right_eye_patches)):
                output_basename = os.path.splitext(entry.name)[0]
                output_left_eye_patch_path = os.path.join(current_left_output_folder,
                                                          f"{output_basename}_left_eye_{idx}.png")
                output_right_eye_patch_path = os.path.join(current_right_output_folder,
                                                           f"{output_basename}_right_eye_{idx}.png")

                cv2.imwrite(output_left_eye_patch_path, left_eye_patch)
                cv2.imwrite(output_right_eye_patch_path, right_eye_patch)

        elif entry.is_dir():
            process_images(input_folder, left_output_folder, right_output_folder,
                           os.path.join(current_folder, entry.name))

input_folder = 'C:\\Users\\jrmun\\Desktop\\Eye_chimeraToPublish'
left_output_folder = 'C:\\Users\\jrmun\\Desktop\\train_left'
right_output_folder = 'C:\\Users\\jrmun\\Desktop\\train_right'
process_images(input_folder, left_output_folder, right_output_folder)

