import os
import shutil
from random import sample


def create_test_sets(left_input_folder, right_input_folder, left_output_folder, right_output_folder, test_ratio=0.5):
    for input_folder, output_folder in [(left_input_folder, left_output_folder), (right_input_folder, right_output_folder)]:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        subfolders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

        for subfolder in subfolders:
            input_subfolder = os.path.join(input_folder, subfolder)
            output_subfolder = os.path.join(output_folder, subfolder)

            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            all_files = [f for f in os.listdir(input_subfolder) if os.path.isfile(os.path.join(input_subfolder, f))]
            test_files_count = int(len(all_files) * test_ratio)
            test_files = sample(all_files, test_files_count)

            for test_file in test_files:
                src_file = os.path.join(input_subfolder, test_file)
                dst_file = os.path.join(output_subfolder, test_file)
                shutil.move(src_file, dst_file)


left_input_folder = "C:\\Users\\jrmun\\Desktop\\train_left"
right_input_folder = "C:\\Users\\jrmun\\Desktop\\train_right"
left_output_folder = "C:\\Users\\jrmun\\Desktop\\test_left"
right_output_folder = "C:\\Users\\jrmun\\Desktop\\test_right"

create_test_sets(left_input_folder, right_input_folder, left_output_folder, right_output_folder)
