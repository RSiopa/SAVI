import numpy as np
import cv2


def main():

    image = cv2.imread("../Aula2/scene.jpg")
    template = cv2.imread("../Aula2/wally.png")

    image2 = image.copy()

    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    top_left = max_lo
    h, w, _ = template.shape

    bottom_right = (top_left[0] + w, top_left[1] + h)

    # cv2.circle(image2, (top_left[0] + w/2, top_left[1] + h/2), round(((top_left[0] - bottom_right[0]) ** 2 + (top_left[1] - bottom_right[1]) ** 2) ** 0.5), 'b', 3)

    cv2.rectangle(image2, top_left, bottom_right, 255, 2)

    cv2.imshow('window2', image2)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
