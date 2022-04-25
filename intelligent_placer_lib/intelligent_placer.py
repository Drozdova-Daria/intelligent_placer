import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
from itertools import permutations
from copy import deepcopy


def in_object(x, y, object_):
    """
    Checking whether a point (x, y) to shape
    :param x: point x-coordinate
    :param y: point y-coordinate
    :param object_: shape
    :return: bool value of belonging of figure
    """
    c = 0
    x_object = [point[0] for point in object_]
    y_object = [point[1] for point in object_]

    for i in range(len(x_object)):
        if (((y_object[i] <= y < y_object[i - 1]) or (y_object[i - 1] <= y < y_object[i]))
                and (x > (x_object[i - 1] - x_object[i]) * (y - y_object[i]) / (y_object[i - 1] - y_object[i]) +
                     x_object[i])):
            c = 1 - c

    return c


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
        approx = cv2.approxPolyDP(contour, 0.009 * epsilon, True)
        objects_contours.append([point[0] for point in approx])

    return objects_contours


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


def get_draw_contour(contour):
    """
    Get coordinates for drawContours function
    :param contour: contour of object
    :return: new view of contour
    """
    object_ = []

    for i in range(len(contour)):
        object_.append([[contour[i][0], contour[i][1]]])

    return np.array(object_)


def draw_contour(image, contour, color):
    """
    Draw contour of shape
    :param image: image
    :param contour: contour of shape
    :param color: contour color
    """
    cv2.drawContours(image, [get_draw_contour(contour)], -1, color, 5)


def get_position_object(image, figure, object_, placed_objects):
    """
    Position an object in a shape
    :param image: image
    :param figure: figure
    :param object_: object
    :param placed_objects: objects located in the figure
    :return: is it possible to position object in figure, object's coordinates in figure
    """
    top_figure, left_figure, right_figure, bottom_figure = get_extreme_points(figure)
    top_object, left_object, right_object, bottom_object = get_extreme_points(object_)

    if (bottom_figure[1] - top_figure[1]) - (bottom_object[1] - top_object[1]) < 0 or \
            (right_figure[0] - left_figure[0]) - (right_object[0] - left_object[0]) < 0:
        return False, []

    top_shift = top_object[1] - top_figure[1]
    left_shift = left_object[0] - left_figure[0]
    for i in range(len(object_)):
        object_[i][0] -= left_shift
        object_[i][1] -= top_shift

    while bottom_object[1] < bottom_figure[1]:
        while right_object[0] < right_figure[0]:
            check = True

            for point in object_:
                if not in_object(point[0], point[1], figure):
                    check = False
                    break
                for placed_object in placed_objects:
                    if in_object(point[0], point[1], placed_object):
                        check = False
                        break
                    for placed_object_point in placed_object:
                        if in_object(placed_object_point[0], placed_object_point[1], object_):
                            check = False
                            break

            if check:
                for placed_object in placed_objects:
                    contours = [get_draw_contour(object_),  get_draw_contour(placed_object)]
                    blank = np.zeros(image.shape[0:2])

                    image_object = cv2.drawContours(blank.copy(), contours, 0, 1)
                    image_placed_object = cv2.drawContours(blank.copy(), contours, 1, 1)
                    if np.logical_and(image_placed_object, image_object).any():
                        check = False
                        break

            if check:
                return True, object_

            for i in range(len(object_)):
                object_[i][0] += 1

        left_shift = left_object[0] - left_figure[0]
        for i in range(len(object_)):
            object_[i][1] += 1
            object_[i][0] -= left_shift

    return False, []


def placed_all(figure, objects, image):
    """
    Make a search for all permutations of objects
    :param figure: figure
    :param objects: objects
    :param image: image
    :return: is it possible to position all objects in figure, objects coordinates in figure
    """
    placed_objects = []
    objects_permutations = list(permutations(objects))

    for objects_ in objects_permutations:
        all_places = True
        for object_ in objects_:
            placed, object_position = get_position_object(image.copy(), figure, deepcopy(object_), placed_objects)

            if not placed:
                all_places = False
                break
            placed_objects.append(object_position)
        if all_places:
            return True, placed_objects

    return False, []


def intelligent_placer(image_name):
    """
    Calling the location function and drawing contours
    :param image_name: image name
    :return: True/False is it possible to position all objects in figure, else None
    """
    if not os.path.exists(image_name):
        print('File error: file ' + image_name + ' doesn\'t exists')
        return None

    image = cv2.imread(image_name)
    if image is None:
        print('File error: unable to open file')
        return None

    objects_contours = get_objects_contours(image)
    figure, objects = classify_objects(objects_contours)

    if len(figure) > 6:
        print('Data error: incorrect number of vertices of figure')
        return None

    if not objects:
        print('Data error: no objects found')
        return None

    placed, placed_objects = placed_all(figure, objects, image.copy())

    draw_contour(image, figure, (255, 0, 0))
    for object_ in objects:
        draw_contour(image, object_, (0, 255, 0))

    if placed:
        for placed_object in placed_objects:
            draw_contour(image, placed_object, (0, 0, 255))

    plt.imshow(image)
    plt.savefig(image_name + '_' + str(placed) + '.png')

    return placed
