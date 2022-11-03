#!/usr/bin/env python3

from __future__ import print_function

from copy import deepcopy

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
    image1 = cv2.imread("../Aula6/images/machu_pichu/2.png")
    image2 = cv2.imread("../Aula6/images/machu_pichu/1.png")

    sift = cv2.SIFT_create(500)

    window_name1 = 'image_matching'
    cv2.namedWindow(window_name1, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name1, 800, 600)

    window_name2 = 'image_stitched'
    cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name2, 800, 600)

    MIN_MATCH_COUNT = 10

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    image_gui1 = deepcopy(image1)

    image_gray1 = cv2.cvtColor(image_gui1, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(image_gray1, None)

    image_kp1 = cv2.drawKeypoints(image_gui1, kp1, None, flags=0)
    # image_kp1 = cv2.drawKeypoints(image_gui1, kp1, image_gui1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



    image_gui2 = deepcopy(image2)

    image_gray2 = cv2.cvtColor(image_gui2, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(image_gray2, None)

    image_kp2 = cv2.drawKeypoints(image_gui2, kp2, None, flags=0)
    # image_kp2 = cv2.drawKeypoints(image_gui2, kp2, image_gui2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w, c = image1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        image2 = cv2.polylines(image2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    # cv.drawMatchesKnn expects list of lists as matches.
    image3 = cv2.drawMatches(image1, kp1, image2, kp2, good, None, **draw_params)

    image4 = deepcopy(image2)

    # image4[int(dst[0][0][0]):int(dst[0][0][0])+w, int(dst[0][0][1]):int(dst[0][0][1])+h] = image1
    # print(dst)

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------

    cv2.imshow(window_name1, image3)
    cv2.imshow(window_name2, image4)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
