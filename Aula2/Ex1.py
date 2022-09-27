#!/usr/bin/env python3

import numpy as np
import cv2


def main():

    image = cv2.imread("../Aula2/lake.jpg")

    for i in np.arange(1, 0.2, -0.01):

        image2 = image.copy()

        image2[:, int(image.shape[1]/2):] = (image2[:, int(image.shape[1]/2):] * i).astype(np.uint8)

        #cv2.imshow('window', image)
        cv2.imshow('window2', image2)

        cv2.waitKey(50)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()
