#!/usr/bin/env python3

import numpy as np
import cv2


def main():
    print('Hello, world!')
    image = np.ndarray((249, 326), dtype=np.uint8)

    image += 129

    cv2.imshow('window', image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
