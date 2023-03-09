import tensorflow as tf
import numpy as np
import os

# Define the path to the directory containing the participant folders
root_dir = 'C:\\Users\\jrmun\\Desktop\\MPIIGaze\\Data\\Extracted'

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

        # Load the annotations from the file for this day
        with open(annot_file, 'r') as f:
            annotations = [line.strip().split() for line in f]

        # Create lists to store the x, y, and image data
        x_data = []
        y_data = []
        img_data = []

        # Create a counter for the image files
        img_counter = 1

        # Define the minimum width threshold
        min_width = 62

        # Iterate through the annotations and load the corresponding image for this day
        for annot in annotations:
            # Extract the x and y values
            x = float(annot[24])
            y = float(annot[25])
            # Load the corresponding image based on the file name
            img_path = os.path.join(img_dir,
                                    f'{img_counter:04d}.jpg')  # Assumes images are named "0001.jpg", "0002.jpg", etc.
            if os.path.exists(img_path):
                img = tf.keras.preprocessing.image.load_img(img_path)
                # Check if the image width is larger than the threshold
                if img.size[0] >= min_width:
                    img = img.resize((224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                    img_data.append(img_array)
                    # Append the x and y values to the data lists
                    x_data.append(x)
                    y_data.append(y)
                else:
                    print(f'Skipping image {img_path} because its width is too small')
            else:
                print(f'Could not find image file: {img_path}')
            # Increment the image file counter
            img_counter += 1

        # Convert the data to NumPy arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        img_data = np.array(img_data)

        # Skip days with insufficient data
        if len(img_data) < 2:
            print(f'Not enough data to train for day {day_name}')
            continue

        # Define the CNN model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2) # Two nodes for x and y values
])

        # Compile the model
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mae', 'mse'])

        # Train the model
        model.fit(img_data, [x_data, y_data], epochs=10, validation_split=0.2)

        # Save the model weights and architecture
        model.save(os.path.join(participant_dir, day_name, 'model.h5'))

        print(f'Done processing day {day_name}.\n')

    print(f'Done processing participant {participant_name}.\n')


