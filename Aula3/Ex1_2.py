#!/usr/bin/env python3

from __future__ import print_function
import cv2
import argparse
import numpy as np


def main():

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    video = cv2.VideoCapture("../Aula3/traffic.mp4")

    rects = [{'name': 'r1', 'x1': 250, 'y1': 500, 'x2': 400, 'y2': 700, 'ncars': 0, 'tic_since_car_count': -500},
         {'name': 'r2', 'x1': 450, 'y1': 500, 'x2': 600, 'y2': 700, 'ncars': 0, 'tic_since_car_count': -500},
         {'name': 'r3', 'x1': 700, 'y1': 500, 'x2': 850, 'y2': 700, 'ncars': 0, 'tic_since_car_count': -500},
         {'name': 'r4', 'x1': 930, 'y1': 500, 'x2': 1080, 'y2': 680, 'ncars': 0, 'tic_since_car_count': -500}]

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    is_first_time = True
    while True:

        ret, frame = video.read()

        if frame is None:
            cv2.waitKey(0)
            break

        stamp = float(video.get(cv2.CAP_PROP_POS_MSEC)) / 1000

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for rect in rects:

            total = 0
            number_of_pixels = 0

            for row in range(rect['y1'], rect['y2']):
                for col in range(rect['x1'], rect['x2']):
                    number_of_pixels += 1
                    total += frame_gray[row, col]

            rect['avg_color'] = int(total / number_of_pixels)

            if is_first_time:
                rect['model_avg_color'] = rect['avg_color']

            diff = abs(rect['avg_color'] - rect['model_avg_color'])

            blackout_time = 1

            if diff > 20 and (stamp - rect['tic_since_car_count']) > blackout_time:
                rect['ncars'] = rect['ncars'] + 1
                rect['tic_since_car_count'] = stamp

        is_first_time = False

        for rect in rects:

            cv2.rectangle(frame, (rect['x1'], rect['y1']), (rect['x2'], rect['y2']), (0, 255, 0), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (0, 255, 0)
            thickness = 1
            text = 'avg=' + str(rect['avg_color']) + ' m=' + str(rect['model_avg_color'])
            frame = cv2.putText(frame, text, (rect['x1'], rect['y1'] - 10), font, fontScale, color, thickness, cv2.LINE_AA)

            text2 = 'ncars= ' + str(rect['ncars'])
            frame = cv2.putText(frame, text2, (rect['x1'], rect['y1'] - 25), font, fontScale, (255, 0, 0), thickness, cv2.LINE_AA)

            text2 = 'Time since lcc= ' + str(round(rect['tic_since_car_count']))
            frame = cv2.putText(frame, text2, (rect['x1'], rect['y1'] - 40), font, fontScale, (255, 0, 0), thickness, cv2.LINE_AA)

        cv2.imshow('window', frame)
        cv2.waitKey(25)

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------

if __name__ == "__main__":
    main()