import numpy as np
import cv2


def main():

    image = cv2.imread("../Aula2/scene.jpg")
    template = cv2.imread("../Aula2/wally.png")
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image2 = image.copy()

    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    H, W, _ = image.shape
    h, w, _ = template.shape

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    mask = np.zeros((H, W)).astype(np.uint8)
    # cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    cv2.circle(mask, (int(top_left[0] + w/2), int(top_left[1] + h/2)), round(((top_left[0] - bottom_right[0]) ** 2 + (top_left[1] - bottom_right[1]) ** 2) ** 0.5), 255, -1)

    mask_bool = mask.astype(bool)

    image_gray_merged = cv2.merge([image_gray, image_gray, image_gray])

    image_gray_merged[mask_bool] = image2[mask_bool]


    cv2.imshow('window', image_gray_merged)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()