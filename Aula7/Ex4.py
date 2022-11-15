#!/usr/bin/env python3
import pickle
from copy import deepcopy
from random import randint, uniform
import cv2
import matplotlib.pyplot as plt
import numpy as np
from models2 import Line
from scipy.optimize import least_squares


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
    line = Line(pts)

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    # Set new values
    line.randomize_Params()

    result = least_squares(line.objectiveFunction, [line.m, line.b], verbose=2)

    line.draw()
    plt.show()

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------


if __name__ == "__main__":
    main()
