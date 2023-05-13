import cv2
import dlib
import os
import time

output_folder = "C:\\Users\\jrmun\\Desktop\\eval2"

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

# Create folders for each direction
for direction in directions:
    os.makedirs(os.path.join(output_folder, direction), exist_ok=True)

def extract_image(frame):
    return frame

def save_image_direction(direction, image):
    # Get the mirrored direction from the mapping
    mirrored_direction = mirrored_directions[direction]

    save_path = os.path.join(output_folder, mirrored_direction)
    file_count = len(os.listdir(save_path))
    cv2.imwrite(os.path.join(save_path, f"{mirrored_direction}_{file_count + 1}.jpg"), image)

def display_countdown(frame, seconds_left):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Capturing in {seconds_left} seconds"
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    x = (frame.shape[1] - text_size[0]) // 2
    y = (frame.shape[0] + text_size[1]) // 2

    cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)

images_per_direction = 1
rounds = 20  # Number of rounds to capture image per direction

for direction in directions:
    print(f"Look at the {direction} direction.")

    for _ in range(rounds):  # Adding a loop to iterate for 20 rounds
        countdown_start = time.time()
        countdown_seconds = 5
        image_count = 0

        while image_count < images_per_direction:
            # Capture frame from the webcam
            ret, frame = cap.read()

            if not ret:
                break

            # Display the countdown timer
            seconds_left = countdown_seconds - int(time.time() - countdown_start)
            if seconds_left > 0:
                display_countdown(frame, seconds_left)
            else:
                countdown_start = time.time()  # Reset the countdown
                # Detect faces in the frame
                faces = face_detector(frame)

                # If a face is detected, find facial landmarks
                if len(faces) > 0:
                    face = faces[0]
                    landmarks = landmark_predictor(frame, face)

                    image = extract_image(frame)

                    save_image_direction(direction, image)
                    image_count += 1

            # Display the direction to look at
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Look {direction}", (10, 50), font, 1, (255, 255, 255), 2)

            # Display the frame
            cv2.imshow("Direction Gaze Calibration", frame)
            key = cv2.waitKey(1) & 0xFF

            # Exit the loop if 'q' key is pressed
            if key == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()

