#!/usr/bin/env python3

from __future__ import print_function
from copy import deepcopy
from random import randint
import cv2
import numpy as np
from scipy.optimize import least_squares

from models import ImageMosaic


def main():

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    # Start video
    q_image = cv2.imread("images/machu_pichu/query_warped.png")
    window_name1 = 'Query Image'
    cv2.namedWindow(window_name1, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name1, 800, 600)

    t_image = cv2.imread("images/machu_pichu/target.png")
    window_name2 = 'Target Image'
    cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name2, 800, 600)

    image_mosaic = ImageMosaic(q_image, t_image)

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    x0 = [image_mosaic.q_scale, image_mosaic.q_bias, image_mosaic.t_scale, image_mosaic.t_bias]
    result = least_squares(image_mosaic.objectiveFunction, x0, verbose=2)

    image_mosaic.draw()
    cv2.waitKey(0)

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------


if __name__ == "__main__":
    main()
