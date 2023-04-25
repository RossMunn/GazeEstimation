import cv2
import dlib
import os
import time

output_folder = "C:\\Users\\jrmun\\Desktop\\test"

# Initialize Dlib's face detector and facial landmarks predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("C:\\Users\\jrmun\\PycharmProjects\\Disso\\extractorModels\\shape_predictor_68_face_landmarks.dat")

# Open the webcam
cap = cv2.VideoCapture(0)

directions = ["00.Centre", "01.UpRight", "02.UpLeft", "03.Right", "04.Left", "05.DownRight", "06.DownLeft"]

# Create folders for each direction
for direction in directions:
    os.makedirs(os.path.join(output_folder, direction), exist_ok=True)

def extract_left_eye(frame, landmarks, padding_ratio=0.2):
    # Left eye landmarks (from 36 to 41)
    points = landmarks.parts()[36:42]
    x_coords = [point.x for point in points]
    y_coords = [point.y for point in points]

    # Calculate the bounding rectangle
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Calculate padding
    width = x_max - x_min
    height = y_max - y_min
    padding_x = int(padding_ratio * width)
    padding_y = int(padding_ratio * height)

    # Add padding to the bounding rectangle
    x_min = max(x_min - padding_x, 0)
    x_max = min(x_max + padding_x, frame.shape[1])
    y_min = max(y_min - padding_y, 0)
    y_max = min(y_max + padding_y, frame.shape[0])

    # Crop the left eye region with padding
    left_eye = frame[y_min:y_max, x_min:x_max]

    return left_eye

def save_image(direction, image):
    save_path = os.path.join(output_folder, direction)
    file_count = len(os.listdir(save_path))
    cv2.imwrite(os.path.join(save_path, f"{direction}_{file_count + 1}.jpg"), image)

def display_countdown(frame, seconds_left):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Capturing in {seconds_left} seconds"
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    x = (frame.shape[1] - text_size[0]) // 2
    y = (frame.shape[0] - text_size[1]) // 2

    cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)

images_per_direction = 32

for direction in directions:
    print(f"Look at the {direction} direction.")

    countdown_start = time.time()
    countdown_seconds = 5
    image_count = 0

    while image_count < images_per_direction:
        # Capture frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Display the direction to look at
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Look {direction}", (10, 50), font, 1, (255, 255, 255), 2)

        # Display the countdown timer
        seconds_left = countdown_seconds - int(time.time() - countdown_start)
        if seconds_left > 0:
            display_countdown(frame, seconds_left)
        else:
            # Detect faces in the frame
            faces = face_detector(frame)

            # If a face is detected, find facial landmarks
            if len(faces) > 0:
                face = faces[0]
                landmarks = landmark_predictor(frame, face)

                left_eye_image = extract_left_eye(frame, landmarks)

                save_image(direction, left_eye_image)
                image_count += 1

                if image_count == images_per_direction:
                    break

        # Display the frame
        cv2.imshow("Direction Gaze Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        # Exit the loop if 'q' key is pressed
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()




