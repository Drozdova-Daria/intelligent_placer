import cv2
from matplotlib import pyplot as plt
import numpy as np


def get_objects_contours(image):
    """
    Find contours of objects in the image
    :param image: original image
    :return: array of object coordinates
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(hsv, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    objects_contours = []
    for contour in contours:
        epsilon = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * epsilon, True)
        objects_contours.append([point[0] for point in approx])

    return objects_contours


def get_extreme_points(coordinates):
    """
    Find extreme points of the object
    :param coordinates:
    :return: top, left, right, bottom - extreme points of object
    """
    top = left = right = bottom = coordinates[0]

    for coordinate in coordinates[1:]:
        if coordinate[0] < left[0]:
            left = coordinate
        elif coordinate[0] > right[0]:
            right = coordinate
        if coordinate[1] < top[1]:
            top = coordinate
        elif coordinate[1] > bottom[1]:
            bottom = coordinate

    return top, left, right, bottom


def in_figure(x, y, x_figure, y_figure):
    """
    Checking whether a point (x, y) to shape
    :param x: point x-coordinate
    :param y: point y-coordinate
    :param x_figure: x-coordinates of figure
    :param y_figure: y-coordinates of figure
    :return: bool value of belonging of figure
    """
    c = 0
    for i in range(len(x_figure)):
        if (((y_figure[i] <= y < y_figure[i - 1]) or (y_figure[i - 1] <= y < y_figure[i]))
                and (x > (x_figure[i - 1] - x_figure[i]) * (y - y_figure[i]) / (y_figure[i - 1] - y_figure[i]) +
                     x_figure[i])):
            c = 1 - c

    return c


def classify_objects(objects_contours):
    """
    Divides objects into figure and objects
    :param objects_contours: array of objects contours
    :return: figure, objects
    """
    figure = 0
    min_coordinate = objects_contours[0][0][0]

    for i in range(len(objects_contours)):
        for point in objects_contours[i]:
            if point[0] < min_coordinate:
                figure = i
                min_coordinate = point[0]

    return objects_contours[figure], objects_contours[:figure] + objects_contours[figure + 1:]


def check_location(figure, object_):
    """
    Check: is it possible to place object_ in a shape
    :param figure: figure contour
    :param object_: object contour
    :return: bool, array: result of check and contour of object if shape can be placed in figure else empty array
    """

    top_figure, left_figure, right_figure, bottom_figure = get_extreme_points(figure)
    top_object, left_object, right_object, bottom_object = get_extreme_points(object_)

    figure_x = [point[0] for point in figure]
    figure_y = [point[1] for point in figure]

    if (bottom_figure[1] - top_figure[1]) - (bottom_object[1] - top_object[1]) < 0 or \
            (right_figure[0] - left_figure[0]) - (right_object[0] - left_object[0]) < 0:
        return False, []

    top_shift = top_object[1] - top_figure[1]
    left_shift = left_object[0] - left_figure[0]
    for i in range(len(object_)):
        object_[i][0] -= left_shift
        object_[i][1] -= top_shift

    for point in object_:
        if not in_figure(point[0], point[1], figure_x, figure_y):
            break

    while bottom_object[1] < bottom_figure[1]:
        while right_object[0] < right_figure[0]:
            check = True
            for point in object_:
                if not in_figure(point[0], point[1], figure_x, figure_y):
                    check = False
                    break
            if check:
                return True, object_
            for i in range(len(object_)):
                object_[i][0] += 1
            points = []
            for point in object_:
                points.append([[point[0], point[1]]])

        left_shift = left_object[0] - left_figure[0]
        for i in range(len(object_)):
            object_[i][1] += 1

            object_[i][0] -= left_shift

    return False, []


def draw_contour(image, contour, color):
    """
    Draw contour of shape
    :param image: image
    :param contour: contour of shape
    :param color: contour color
    """
    object_ = []

    for i in range(len(contour)):
        object_.append([[contour[i][0], contour[i][1]]])

    cv2.drawContours(image, [np.array(object_)], -1, color, 5)


def intelligent_placer(image, image_name, results_path):
    """
    Calling functions
    :param image: image
    :param image_name: image name
    :param results_path: path for result
    """
    objects_contours = get_objects_contours(image)
    figure, objects = classify_objects(objects_contours)

    for j in range(len(objects)):
        image_ = image.copy()

        draw_contour(image_, objects[j], (255, 0, 0))
        draw_contour(image_, figure, (0, 255, 0))

        check, coordinates = check_location(figure, objects[j].copy())

        if check:
            draw_contour(image_, coordinates, (0, 0, 255))

        plt.imshow(image_)
        plt.savefig(results_path + image_name + '_' + str(j) + '.jpg')


