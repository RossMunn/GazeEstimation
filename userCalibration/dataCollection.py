import cv2
import dlib
import os
import time

output_folder = "C:\\Users\\jrmun\\Desktop\\Calibration_data"

# Initialize Dlib's face detector and facial landmarks predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("C:\\Users\\jrmun\\PycharmProjects\\Disso\\extractorModels\\shape_predictor_68_face_landmarks.dat")

# Open the webcam
cap = cv2.VideoCapture(0)

directions = ["00.Centre", "01.UpRight", "02.UpLeft", "03.Right", "04.Left", "05.DownRight", "06.DownLeft"]

# Create a mapping for mirrored directions
mirrored_directions = {
    "01.UpRight": "02.UpLeft",
    "02.UpLeft": "01.UpRight",
    "03.Right": "04.Left",
    "04.Left": "03.Right",
    "05.DownRight": "06.DownLeft",
    "06.DownLeft": "05.DownRight",
    "00.Centre": "00.Centre",
}

# Create folders for each direction and eye
for eye in ["left", "right"]:
    for direction in directions:
        os.makedirs(os.path.join(output_folder, eye, direction), exist_ok=True)

def extract_eye(frame, landmarks, eye_points, padding_ratio=0.2):
    points = [landmarks.part(point) for point in eye_points]
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

    # Crop the eye region with padding
    eye = frame[y_min:y_max, x_min:x_max]

    return eye

def save_image(eye, direction, image):
    # Get the mirrored direction from the mapping
    mirrored_direction = mirrored_directions[direction]

    save_path = os.path.join(output_folder, eye, mirrored_direction)
    file_count = len(os.listdir(save_path))
    cv2.imwrite(os.path.join(save_path, f"{mirrored_direction}_{file_count + 1}.jpg"), image)

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

                left_eye_image = extract_eye(frame, landmarks, list(range(36, 42)))
                right_eye_image = extract_eye(frame, landmarks, list(range(42, 48)))

                save_image("left", direction, left_eye_image)
                save_image("right", direction, right_eye_image)
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

