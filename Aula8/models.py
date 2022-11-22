#!/usr/bin/env python3
import math
import pickle
from copy import deepcopy
from random import randint, uniform
import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImageMosaic:
    """Defines the model of a line segment"""

    def __init__(self, q_image, t_image):

        self.q_image = q_image
        self.t_image = t_image

        # Make sure we are using float images
        self.q_image = self.q_image.astype(float) / 255.0
        self.t_image = self.t_image.astype(float) / 255.0

        self.q_height, self.q_width, _ = q_image.shape
        self.t_height, self.t_width, _ = t_image.shape

        self.mask = q_image[:, :, 0] > 0
        self.randomize_Params()

        cv2.namedWindow('stitched_image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('stitched_image', 800, 600)

        stitched_image_f = deepcopy(self.t_image)
        stitched_image_f[self.mask] = (self.q_image[self.mask] + stitched_image_f[self.mask]) / 2
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Original', 800, 600)
        self.drawFloatImage('Original', stitched_image_f)

    def randomize_Params(self):

        self.q_scale = 1.0
        self.q_bias = 0.0
        self.t_scale = 1.0
        self.t_bias = 0.0

    def correctImage(self, image, scale, bias):

        image_c = scale * image + bias
        image_c[image_c > 1] = 1  # Saturate at 1
        image_c[image_c < 0] = 0  # Undersaturate at 0
        return image_c

    def objectiveFunction(self, params):

        self.q_scale = params[0]
        self.q_bias = params[1]
        self.t_scale = params[2]
        self.t_bias = params[3]

        # Correct images with the parameters
        self.q_image_c = self.correctImage(self.q_image, self.q_scale, self.q_bias)
        self.t_image_c = self.correctImage(self.t_image, self.t_scale, self.t_bias)

        residuals = []

        diffs = np.abs(self.t_image_c - self.q_image_c)
        diffs_in_overlap = diffs[self.mask]
        residuals = np.sum(diffs_in_overlap)

        print('residuals=' + str(residuals))

        self.draw()

        return residuals

    def drawFloatImage(self, win_name, image_f):
        image_uint8 = (image_f*255).astype(np.uint8)
        cv2.imshow(win_name, image_uint8)

    def drawBooleanImage(self, win_name, image_f):
        image_uint8 = (image_f*255).astype(np.uint8)
        cv2.imshow(win_name, image_uint8)

    def draw(self):

        stitched_image_f = deepcopy(self.t_image_c)
        stitched_image_f[self.mask] = (self.q_image_c[self.mask] + stitched_image_f[self.mask]) / 2
        self.drawFloatImage('stitched_image', stitched_image_f)

        print('drawing images')
        # cv2.waitKey(1)
