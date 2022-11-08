#!/usr/bin/env python3

from __future__ import print_function

from copy import deepcopy
from random import randint

import cv2
import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt


def main():

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    # Start video
    image1 = cv2.imread("../Aula6/images/castle/1.png")
    image2 = cv2.imread("../Aula6/images/castle/original.jpg")

    sift = cv2.SIFT_create(500)

    window_name1 = 'image_matching'
    cv2.namedWindow(window_name1, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name1, 800, 600)

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    image_gui1 = deepcopy(image1)

    image_gray1 = cv2.cvtColor(image_gui1, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(image_gray1, None)

    # image_kp1 = cv2.drawKeypoints(image_gui1, kp1, None, flags=0)
    for idx, key_point in enumerate(kp1):
        x1 = int(key_point.pt[0])
        y1 = int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.circle(image_gui1, (x1, y1), 50, color, 3)


    image_gui2 = deepcopy(image2)

    image_gray2 = cv2.cvtColor(image_gui2, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(image_gray2, None)

    # image_kp2 = cv2.drawKeypoints(image_gui2, kp2, None, flags=0)
    for idx, key_point in enumerate(kp2):
        x2 = int(key_point.pt[0])
        y2 = int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.circle(image_gui2, (x2, y2), 50, color, 3)


    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    # matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    image3 = cv2.drawMatches(image1, kp1, image2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------

    cv2.imshow(window_name1, image3)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
