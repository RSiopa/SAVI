#!/usr/bin/env python3
import math
import pickle
from copy import deepcopy
from random import randint, uniform
import cv2
import matplotlib.pyplot as plt
import numpy as np


class Sinusoid:
    """Defines the model of a sinusoidal function"""

    def __init__(self, gt):

        self.gt = gt
        self.randomize_Params()
        self.first_draw = True
        self.xs_for_plot = list(np.linspace(-10, 10, num=500))

    def randomize_Params(self):
        self.a = uniform(-10, 10)
        self.b = uniform(-10, 10)
        self.h = uniform(-10, 10)
        self.k = uniform(-10, 10)

    def getY(self, x):
        return self.a * math.sin(self.b * (x - self.h)) + self.k

    def getYs(self, xs):
        ys = []
        for x in xs:
            ys.append(self.getY(x))
        return ys

    def objectiveFunction(self):
        residuals = []

        for gt_x, gt_y in zip(self.gt['xs'], self.gt['ys']):
            y = self.getY(gt_x)
            residual = abs(y - gt_y)
            residuals.append(residual)

        # Error is the sum of the residuals
        error = sum(residuals)
        return error

    def draw(self, color='b--'):
        xi = -10
        xf = 10
        yi = self.getY(xi)
        yf = self.getY(xf)

        if self.first_draw:
            self.draw_handle = plt.plot(self.xs_for_plot, self.getYs(self.xs_for_plot), color, linewidth=2)
            self.first_draw = False
        else:
            plt.setp(self.draw_handle, data=(self.xs_for_plot, self.getYs(self.xs_for_plot)))


class Line:
    """Defines the model of a line segment"""

    def __init__(self, gt):

        self.gt = gt
        self.randomize_Params()
        self.first_draw = True

    def randomize_Params(self):
        self.m = uniform(-2, 2)
        self.b = uniform(-5, 5)

    def getY(self, x):
        return self.m * x + self.b

    def objectiveFunction(self):

        residuals = []

        for gt_x, gt_y in zip(self.gt['xs'], self.gt['ys']):
            y = self.getY(gt_x)
            residual = abs(y - gt_y)
            residuals.append(residual)

        # Error is the sum of the residuals
        error = sum(residuals)
        return error

    def draw(self, color='b--'):
        xi = -10
        xf = 10
        yi = self.getY(xi)
        yf = self.getY(xf)

        if self.first_draw:
            self.draw_handle = plt.plot([xi, xf], [yi, yf], color, linewidth=2)
            self.first_draw = False
        else:
            plt.setp(self.draw_handle, data=([xi, xf], [yi, yf]))
