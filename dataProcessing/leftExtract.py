import cv2
import os
import shutil

# Define the path to the directory containing the participant folders
root_dir = 'C:\\Users\\jrmun\\Desktop\\MPIIGaze\\Data\\Original'

# Define the path to the output directory for extracted images
output_dir = 'C:\\Users\\jrmun\\Desktop\\MPIIGaze\\Data\\Extracted'

# Define the path to the haarcascade file
cascade_file = 'C:\\Users\\jrmun\\PycharmProjects\\Disso\\haarcascadeXML\\haarcascade_mcs_lefteye.xml'

# Load the haarcascade classifier
cascade_classifier = cv2.CascadeClassifier(cascade_file)

# Loop through each participant folder
for participant_name in os.listdir(root_dir):
    # Define the path to the participant folder
    participant_dir = os.path.join(root_dir, participant_name)
    if not os.path.isdir(participant_dir):
        continue

    print(f'Processing participant {participant_name}...')

    # Loop through each day folder in the participant folder
    for day_name in os.listdir(participant_dir):
        # Skip the calibration folder
        if day_name == 'Calibration':
            continue

        print(f'Processing day {day_name}...')

        # Define the path to the directory containing the images for this day
        img_dir = os.path.join(participant_dir, day_name)

        # Define the path to the annotations file for this day
        annot_file = os.path.join(participant_dir, day_name, 'annotation.txt')

        # Create a new directory for the extracted images for this day
        output_day_dir = os.path.join(output_dir, participant_name, day_name)
        os.makedirs(output_day_dir, exist_ok=True)

        # Copy the annotations file to the new directory for this day
        shutil.copy(annot_file, output_day_dir)

        # Load the annotations from the file for this day
        with open(os.path.join(output_day_dir, 'annotation.txt'), 'r') as f:
            annotations = [line.strip().split() for line in f]

        # Create a counter for the image files
        img_counter = 1

        # Iterate through the annotations and extract the left eye images for this day
        for annot in annotations:
            # Load the corresponding image based on the file name
            img_path = os.path.join(img_dir, f'{img_counter:04d}.jpg')
            if not os.path.exists(img_path):
                print(f'Could not find image file: {img_path}')
                continue

            # Load the image and convert to grayscale
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect the eyes in the image using the haarcascade classifier
            eyes = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Extract the left eye image if it exists
            left_eye = None
            for (x, y, w, h) in eyes:
                if x < img.shape[1] // 2:  # Check if the eye is on the left side of the image
                    left_eye = img[y:y+h, x:x+w]
                    break

            if left_eye is not None:
                # Save the left eye image to the output directory for this day
                output_img_path = os.path.join(output_day_dir, f'{img_counter:04d}.jpg')
                cv2.imwrite(output_img_path, left_eye)

                # Increment the image file counter
            img_counter += 1

            print(f'Done processing day {day_name} + {img_counter}.\n')

        print(f'Done processing participant {participant_name}.\n')
