from itertools import count

import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os


class ImgLib:
    def __init__(self):
        self.img_size = 10

    def read_image(img_path):
        return plt.imread(img_path)


    def show_image(image):
        plt.imshow(image)
        plt.show()


    def resize_image(image, image_size):
        return cv2.resize(image, image_size)


    def change_color_space(image, to_color_space):
        if to_color_space == 'GRAY':
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif to_color_space == 'YUV':
            return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif to_color_space == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            print("color space needs to be GRAY or HSV or YUV")


    def aspect_ratio(image):
        height, width, channels = image.shape()
        return width/height


    def image_dimensions(image):
        print("--- image dimensions --- ")
        print("width: ", image.shape[1])
        print("height: ", image.shape[0])
        print("channels: ", image.shape[2])


    def calculate_histogram(image, channel):
        return cv2.calcHist([image], [channel], None, [256], [0, 256])


    def draw_histogram(image, color = 'red'):
        if len(image.shape) == 3:
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
            plt.show()

        else:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.show()




