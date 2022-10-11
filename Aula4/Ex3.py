#!/usr/bin/env python3

from __future__ import print_function

from copy import deepcopy

import cv2
import argparse
import numpy as np
import csv

from functions import Detection, Tracker


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

    person_detector = cv2.CascadeClassifier('haarcascade_fullbody.xml')

    detection_counter = 0
    tracker_counter = 0
    trackers = []
    iou_threshold = 128

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

        image_gray = cv2.cvtColor(image_gui, cv2.COLOR_BGR2GRAY)

        bboxes = person_detector.detectMultiscale(image_gray, scaleFactor = 1.05, minNeighbor = 5, minSize = (20, 40))

        detections = []

        for bbox in bboxes:
            x1, y1, w, h = bbox
            detection = Detection(x1, y1, w, h, image_gray)
            detection_counter += 1
            detection.draw(image_gui)
            detections.append(detection)
            cv2.imshow('detection ' + str(detection.id), detection.image)

            # cv2.rectangle(image_gui, (x1, y1), (x1 + w, y1 + h), (0, 0, 255, 3))

        for detection in detections:
            for tracker in trackers:
                tracker_bbox = tracker.detections[-1]
                iou = detection.computeIOU(tracker_bbox)
                print('IOU( T' + str(tracker.id) + ' D' + str(detection.id) + ' ) = ' + str(iou))
                if iou > iou_threshold:
                    tracker.addDetection(detection)

        for detection in detections:
            tracker = Tracker(detection, id=tracker_counter)
            tracker_counter += 1
            trackers.append(tracker)

        for tracker in trackers:
            tracker.draw(image_gui)

        for tracker in trackers:
            print(tracker)

        cv2.imshow(window_name, image_gui)
        cv2.waitKey(25)

        frame_counter += 1

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------




if __name__ == "__main__":
    main()
