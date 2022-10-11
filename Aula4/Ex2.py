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

    video = cv2.VideoCapture("../Aula4/TownCentreXVID.mp4")

    file = open('TownCentre-groundtruth.top')

    window_name = 'image_rgb'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 500)

    number_of_people = 0

    csv_reader = csv.reader(open('TownCentre-groundtruth.top'))
    for row in csv_reader:

        if len(row) != 12:
            continue

        personNumber, _, _, _, _, _, _, _, _, _, _, _ = row
        personNumber = int(personNumber)

        if personNumber >= number_of_people:
            number_of_people = personNumber + 1

    colors = np.random.randint(0, high=255, size=(number_of_people, 3), dtype=int)

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    frame_counter = 0

    while True:

        ret, frame = video.read()
        image_gui = deepcopy(frame)

        if frame is None:
            cv2.waitKey(0)
            break

        csv_reader = csv.reader(open('TownCentre-groundtruth.top'))

        for row in csv_reader:

            if len(row) != 12:
                continue

            personNumber, frameNumber, _, _, _, _, _, _, bodyLeft, bodyTop, bodyRight, bodyBottom = row

            personNumber = int(personNumber)
            frameNumber = int(frameNumber)
            bodyLeft = int(float(bodyLeft))
            bodyRight = int(float(bodyRight))
            bodyTop = int(float(bodyTop))
            bodyBottom = int(float(bodyBottom))

            if frame_counter != frameNumber:
                continue

            x1 = bodyLeft
            y1 = bodyTop
            x2 = bodyRight
            y2 = bodyBottom
            color = colors[personNumber, :]
            cv2.rectangle(image_gui, (x1, y1), (x2, y2), (int(color[0]), int(color[1]), int(color[2])), 3)


        cv2.imshow(window_name, image_gui)
        cv2.waitKey(25)

        frame_counter += 1

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------


if __name__ == "__main__":
    main()
