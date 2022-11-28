import numpy as np
import cv2


def hard_noise_removing_filter(image, show_preprocess=False):
    """Removes image noise using cv2 and rather heavy methods, possibly
    damaging delicate char parts"""

    # Greyscale image - remove other colors
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if show_preprocess:
        cv2.imshow("gray", gray_image)
        cv2.waitKey()

    # Add pixels to thin places before blurring
    image = cv2.dilate(gray_image, np.ones((2,2), np.uint8), iterations = 1)
    if show_preprocess:
        cv2.imshow("dilation", image)
        cv2.waitKey()

    # Noise removal
    image = cv2.medianBlur(image, 3)
    if show_preprocess:
        cv2.imshow("noise", image)
        cv2.waitKey()

    # Erosion
    image = cv2.erode(image, np.ones((3,3), np.uint8), iterations = 1)
    if show_preprocess:
        cv2.imshow("erosion", image)
        cv2.waitKey()

    # Threshold the image (convert it to pure black and white)
    image = cv2.threshold(image, 0, 230, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if show_preprocess:
        cv2.imshow("thresh", image)
        cv2.waitKey()

    # Erosion
    image = cv2.dilate(image, np.ones((2,2), np.uint8), iterations = 1)
    if show_preprocess:
        cv2.imshow("dilateerosion", image)
        cv2.waitKey()

    return image

