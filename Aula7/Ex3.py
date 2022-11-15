#!/usr/bin/env python3
import pickle
from copy import deepcopy
from random import randint, uniform
import cv2
import matplotlib.pyplot as plt
import numpy as np
from models import Sinusoid


def main():
    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    file = open('pts.pkl', 'rb')
    pts = pickle.load(file)
    file.close()
    print('pts = ' + str(pts))

    plt.figure()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    print('Created a figure')

    plt.plot(pts['xs'], pts['ys'], 'sk', linewidth=2, markersize=6)

    # Define the model
    # y = m * x + b
    model = Sinusoid(pts)
    best_model = Sinusoid(pts)
    best_error = 1E6

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    while True:  # Iterate setting new values for the params and recomputing the error

        # Set new values
        model.randomize_Params()

        error = model.objectiveFunction()

        if error < best_error:
            best_model.a = model.a
            best_model.b = model.b
            best_model.h = model.h
            best_model.k = model.k
            best_error = error

        model.draw()
        best_model.draw('r-')

        plt.waitforbuttonpress(0.1)
        if not plt.fignum_exists(1):
            print('Terminated')
            break


    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------


if __name__ == "__main__":
    main()