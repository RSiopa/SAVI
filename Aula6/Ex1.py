#!/usr/bin/env python3

from __future__ import print_function

from copy import deepcopy

import cv2
import argparse
import numpy as np
import csv


def main():

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    # Start video
    image = cv2.imread("../Aula6/images/santorini/1.png")

    sift = cv2.SIFT_create(500)

    window_name = 'image_sift'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 500)

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    image_gui = deepcopy(image)

    image_gray = cv2.cvtColor(image_gui, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(image_gray, None)

    image_kp = cv2.drawKeypoints(image_gui, kp, image_gui, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------

    cv2.imshow(window_name, image_kp)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
