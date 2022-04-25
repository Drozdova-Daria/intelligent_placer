import cv2
import os

from intelligent_placer import intelligent_placer


def main():
    """
    Calling algorithm for all images of directory
    """
    images_path = 'images'
    results_path = 'results\\'
    for image_name in os.listdir(images_path):
        image = cv2.imread(images_path + '\\' + image_name)
        intelligent_placer(image, image_name, results_path)


if __name__ == '__main__':
    main()
