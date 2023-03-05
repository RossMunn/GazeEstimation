
import os
import shutil
import csv

# Define the paths to the original and labeled data folders
original_data_path = r'C:\\Users\\jrmun\\Desktop\\MPIIGaze\\Data\\Original'
labeled_data_path = r'C:\\Users\\jrmun\\Desktop\\processeddataset'

# Define a dictionary to map label names to folder names
label_folder_map = {
    'Up': 'Up',
    'Down': 'Down',
    'Left': 'Left',
    'Right': 'Right'
}

# Loop over each row in the CSV file
with open('C:\\Users\\jrmun\\Desktop\\MPIIGaze\\Up_Down_Right_Left.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        day = int(float(row[0]))
        image_number = int(float(row[1]))
        label = row[2]
        person = row[3]

        # If the label is not recognized, skip this row
        if label not in label_folder_map:
            print(f'Skipping row with unknown label: {label}')
            continue

        # Define the paths to the original and labeled image files
        original_image_path = os.path.join(original_data_path, person, f'day{day:02d}', f'{image_number:04d}.jpg')
        labeled_image_path = os.path.join(labeled_data_path, label_folder_map[label], f'{person}_day{day:02d}_{image_number:04d}.jpg')

        # Create the folder for the labeled image, if it doesn't already exist
        os.makedirs(os.path.dirname(labeled_image_path), exist_ok=True)

        # Copy the original image to the labeled image folder
        shutil.copy(original_image_path, labeled_image_path)
        print(f'Copied {original_image_path} to {labeled_image_path}')
