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

    person_detector = cv2.CascadeClassifier('../Aula4/haarcascade_fullbody.xml')

    detection_counter = 0
    tracker_counter = 0
    trackers = []
    iou_threshold = 0.8

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

        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bboxes = person_detector.detectMultiScale(image_gray, scaleFactor = 1.2, minNeighbors = 4, minSize = (20, 40))

        detections = []
        for bbox in bboxes:
            x1, y1, w, h = bbox
            detection = Detection(x1, y1, w, h, image_gray, id = detection_counter, stamp = stamp)
            detection_counter += 1
            detection.draw(image_gui)
            detections.append(detection)
            # cv2.imshow('detection ' + str(detection.id), detection.image)

        for detection in detections:
            for tracker in trackers:
                tracker_bbox = tracker.detections[-1]
                iou = detection.computeIOU(tracker_bbox)
                print('IOU( T' + str(tracker.id) + ' D' + str(detection.id) + ' ) = ' + str(iou))
                if iou > iou_threshold:
                    tracker.addDetection(detection, image_gray)

        for tracker in trackers:
            last_detection_id = tracker.detections[-1]
            detection_ids = [d.id for d in detections]
            if not last_detection_id in detection_ids:
                print('Tracker ' + str(tracker.id) + ' doing some tracking')
                tracker.track(image_gray)

        for tracker in trackers:
            print('Tracker ' + str(tracker.id) + ' active for ' + round(stamp-tracker.getLastDetectionStamp()) + ' secs')


        for detection in detections:
            if not detection.assigned_to_tracker:
                tracker = Tracker(detection, id = tracker_counter)
                tracker_counter += 1
                trackers.append(tracker)

        for tracker in trackers:
            tracker.draw(image_gui)

            # window2 = 'T' + str(tracker.id) + ' template'
            # cv2.imshow(window2, tracker.template)

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
