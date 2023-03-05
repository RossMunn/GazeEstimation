import os
import cv2
import shutil

# define the paths to the subfolders containing images
folders_path = 'C:\\Users\\jrmun\\Desktop\\processeddataset'

# define the path to the haar cascade file for detecting eyes
cascade_path = '/haarcascadeXML/haarcascade_eye.xml'

# define the path to the output folder for the extracted left eye images
output_folder_path = 'C:\\Users\\jrmun\\Desktop\\leftEye_Dataset'


# create the output folder if it does not exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# create a face detector object using the haar cascade file
eye_cascade = cv2.CascadeClassifier(cascade_path)

# loop over all subfolders
for folder in os.listdir(folders_path):
    subfolder_path = os.path.join(folders_path, folder)

    # create the corresponding subfolder in the output folder
    output_subfolder_path = os.path.join(output_folder_path, folder)
    if not os.path.exists(output_subfolder_path):
        os.makedirs(output_subfolder_path)

    # loop over all files in the subfolder
    for file_name in os.listdir(subfolder_path):
        image_path = os.path.join(subfolder_path, file_name)

        # read the image
        image = cv2.imread(image_path)

        # detect the eyes in the image
        eyes = eye_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)

        # loop over all detected eyes
        for (x, y, w, h) in eyes:
            # check if the detected eye is on the left side of the face
            if x < image.shape[1] / 2:
                # crop the left eye
                left_eye = image[y:y + h, x:x + w]

                # save the left eye to the corresponding subfolder in the output folder
                output_file_name = 'left_eye_' + file_name
                output_file_path = os.path.join(output_subfolder_path, output_file_name)
                cv2.imwrite(output_file_path, left_eye)
