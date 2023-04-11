import os
import cv2


def flip_and_save_image(img_path, save_path):
    img = cv2.imread(img_path)
    flipped_img = cv2.flip(img, 1)
    cv2.imwrite(save_path, flipped_img)


def flip_images(src_dir, dest_dir):
    opposite_classes = {
        "00.Centre": "00.Centre",
        "01.UpRight": "02.UpLeft",
        "02.UpLeft": "01.UpRight",
        "03.Right": "04.Left",
        "04.Left": "03.Right",
        "05.DownRight": "06.DownLeft",
        "06.DownLeft": "05.DownRight"
    }

    for class_name in opposite_classes:
        src_class_dir = os.path.join(src_dir, class_name)
        dest_class_dir = os.path.join(dest_dir, opposite_classes[class_name])

        if not os.path.exists(dest_class_dir):
            os.makedirs(dest_class_dir)

        for img_name in os.listdir(src_class_dir):
            src_img_path = os.path.join(src_class_dir, img_name)
            dest_img_path = os.path.join(dest_class_dir, img_name)
            flip_and_save_image(src_img_path, dest_img_path)

if __name__ == "__main__":
    source_directory = "C:\\Users\\jrmun\\Desktop\\train_left"
    destination_directory = "C:\\Users\\jrmun\\Desktop\\train_right"
    flip_images(source_directory, destination_directory)


