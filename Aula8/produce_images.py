#!/usr/bin/env python3

from __future__ import print_function
from copy import deepcopy
from random import randint
import cv2
import numpy as np


def main():

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    # Start video
    image1 = cv2.imread("../Aula8/images/machu_pichu/2.png")
    image2 = cv2.imread("../Aula8/images/machu_pichu/1.png")

    # Sift with 500 points
    sift = cv2.SIFT_create(500)

    # Variables
    MIN_MATCH_COUNT = 10

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    image_gui1 = deepcopy(image1)

    # Keypoint acquisition from image 1
    image_gray1 = cv2.cvtColor(image_gui1, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(image_gray1, None)

    # Draw keypoints
    for idx, key_point in enumerate(kp1):
        x1 = int(key_point.pt[0])
        y1 = int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.circle(image_gui1, (x1, y1), 50, color, 3)

    image_gui2 = deepcopy(image2)
    image_stitched = deepcopy(image2)

    # Keypoint acquisition from image 2
    image_gray2 = cv2.cvtColor(image_gui2, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(image_gray2, None)

    # Draw keypoints
    for idx, key_point in enumerate(kp2):
        x2 = int(key_point.pt[0])
        y2 = int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.circle(image_gui2, (x2, y2), 50, color, 3)

    # Matching of images
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test to exclude bad matches
    good = []
    for best_match, second_best_match in matches:
        if best_match.distance < 0.7 * second_best_match.distance:
            good.append(best_match)

    # findHomography/perspectiveTransform
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w, c = image1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # Dray in image2 where image1 is
        image_gui2 = cv2.polylines(image_gui2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    # Params for match drawing
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    # Draws lines from matching
    image3 = cv2.drawMatches(image1, kp1, image2, kp2, good, None, **draw_params)

    # Warp q image to move it to the t image coordinate frame
    q_h, q_w, _ = image1.shape
    t_h, t_w, _ = image2.shape

    # When q_image is inside t_image
    stitched_image_h = t_h
    stitched_image_w = t_w

    q_image_warped = cv2.warpPerspective(image1, M, (stitched_image_w, stitched_image_h))
    q_image_warped = q_image_warped[:, :, 0:3]  # remove fourth channel

    # Change image brightness
    # image_gui3 = image_gui3 - 50

    # Put image1 in image2, and make average of pixels
    image_stitched[int(dst[0][0][1]):int(dst[0][0][1])+h, int(dst[0][0][0]):int(dst[0][0][0])+w] = ((image1.astype(float) + image_stitched[int(dst[0][0][1]):int(dst[0][0][1])+h, int(dst[0][0][0]):int(dst[0][0][0])+w].astype(float))/2).astype(np.uint8)

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------

    cv2.imwrite('images/machu_pichu/query_warped.png', q_image_warped)
    cv2.imwrite('images/machu_pichu/target.png', image_stitched)


if __name__ == "__main__":
    main()
