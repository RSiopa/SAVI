#!/usr/bin/env python3

from copy import deepcopy
from random import randint
import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    coord_dict = {'xs': [], 'ys': []}

    # make data
    x = np.linspace(-10, 10, 2)
    y = np.linspace(-5, 5, 2)

    fig, ax = plt.subplots()

    # ax.plot(x, y, linewidth=2.0)

    cv2.setMouseCallback('image', draw_circle)

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    while True:

        plt.show()


        k = cv2.waitKey(0)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------


# mouse callback function
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        coord_dict['xs'] = coord_dict['xs'].append(x)
        coord_dict['ys'] = coord_dict['ys'].append(y)


if __name__ == "__main__":
    main()
