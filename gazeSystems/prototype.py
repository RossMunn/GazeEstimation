import cv2
import dlib
import imutils
import numpy as np
import os
import queue
import threading
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import SGD
import tkinter as tk

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

def fine_tune_model(model, calibration_dir):
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.3)
    train_data = datagen.flow_from_directory(
        calibration_dir,
        target_size=(50, 42),
        color_mode="grayscale",
        batch_size=6,
        class_mode="categorical",
        subset='training')

    val_data = datagen.flow_from_directory(
        calibration_dir,
        target_size=(50, 42),
        color_mode="grayscale",
        batch_size=6,
        class_mode="categorical",
        subset='validation')

    model.compile(loss="categorical_crossentropy", optimizer=SGD(), metrics=["accuracy"])
    model.fit(train_data, validation_data=val_data, epochs=10)

    return model

# Load left eye model
model_path_left = 'C:\\Users\\jrmun\\PycharmProjects\\Disso\\Models\\best_eye_gaze_model_left.h5'
model_left = load_model(model_path_left)

# Load right eye model
model_path_right = 'C:\\Users\\jrmun\\PycharmProjects\\Disso\\Models\\best_eye_gaze_model_right.h5'
model_right = load_model(model_path_right)

# Fine-tune the models with calibration data
calibration_dir_left = 'C:\\Users\\jrmun\\Desktop\\cali_left'
calibration_dir_right = 'C:\\Users\\jrmun\\Desktop\\cali_right'

model_left = fine_tune_model(model_left, calibration_dir_left)
model_right = fine_tune_model(model_right, calibration_dir_right)

eye_direction_queue = queue.Queue()

def eye_gaze_direction():
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
            elif user_choice == 2:
                score = prediction_right
            elif user_choice == 3:
                score = (prediction_left + prediction_right) / 2
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

            if np.argmax(score) == upleft_index:
                eye_direction_queue.put('Up')
            elif np.argmax(score) == upright_index:
                eye_direction_queue.put('Up')
            elif np.argmax(score) == downleft_index:
                eye_direction_queue.put('Down')
            elif np.argmax(score) == downright_index:
                eye_direction_queue.put('Down')
            elif np.argmax(score) == left_index:
                eye_direction_queue.put('Left')
            elif np.argmax(score) == right_index:
                eye_direction_queue.put('Right')
            elif np.argmax(score) == center_index:
                eye_direction_queue.put('Centre')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

eye_gaze_thread = threading.Thread(target=eye_gaze_direction)
eye_gaze_thread.start()

class Keyboard(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.geometry('1200x600')
        self.grid()
        self.selected_key = [0, 0]  # Initialize selected_key as a list
        self.create_widgets()
        self.focus_set()
        self.after(100, self.check_eye_direction)

    def create_widgets(self):
        self.buttons = []
        keys = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i in range(4):
            row = []
            for j in range(7):
                if 7 * i + j < len(keys):
                    button = tk.Button(self, text=keys[7 * i + j], width=10, height=5)
                    button.grid(row=i, column=j)
                    row.append(button)
            self.buttons.append(row)

    def check_eye_direction(self):
        if not eye_direction_queue.empty():
            eye_direction = eye_direction_queue.get()
            if self.selected_key is not None:
                self.buttons[self.selected_key[0]][self.selected_key[1]].config(relief='raised',bg='SystemButtonFace')  # Reset to default color
            if eye_direction == 'Up':
                self.selected_key[0] = (self.selected_key[0] - 1) % 4
            elif eye_direction == 'Down':
                self.selected_key[0] = (self.selected_key[0] + 1) % 4
            elif eye_direction == 'Left':
                self.selected_key[1] = (self.selected_key[1] - 1) % 7
            elif eye_direction == 'Right':
                self.selected_key[1] = (self.selected_key[1] + 1) % 7
            elif eye_direction == 'Centre':
                print(self.buttons[self.selected_key[0]][self.selected_key[1]].cget('text'))
            self.buttons[self.selected_key[0]][self.selected_key[1]].config(relief='sunken',bg='green')  # Change the color of the selected button

        self.after(100, self.check_eye_direction)

root = tk.Tk()
keyboard = Keyboard(master=root)
keyboard.mainloop()

# Stop the eye gaze thread when the GUI is closed
eye_gaze_thread.join()


