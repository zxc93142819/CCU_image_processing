import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

def sobel(img):
    neighbor = np.array([[-1, -1], [0, -1], [1, -1],
                         [-1, 0], [0, 0], [1, 0],
                         [-1, 1], [0, 1], [1, 1]])
    sx = np.array([-1, 0, 1,
                   -2, 0, 2,
                   -1, 0, 1])
    sy = np.array([-1, -2, -1,
                    0, 0, 0,
                    1, 2, 1])
    row, col, channel = img.shape[0], img.shape[1], img.shape[2]
    img_sob = np.zeros((row, col, channel))
    img = img / 255.0
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            for k in range(0, channel):
                gx = 0
                gy = 0
                for m in range(0, 9):
                    r = i + neighbor[m][0]
                    c = j + neighbor[m][1]
                    gx = gx + img[r][c][k] * sx[m]
                    gy = gy + img[r][c][k] * sy[m]
                img_sob[i][j][k] = abs(gx) + abs(gy)
    return img_sob

img1 = cv2.imread('./HW4_test_image/image1.jpg')
img2 = cv2.imread('./HW4_test_image/image2.jpg')
img3 = cv2.imread('./HW4_test_image/image3.jpg')

img1_sob = sobel(img1)
img2_sob = sobel(img2)
img3_sob = sobel(img3)

cv2.imshow('img1', img1)
cv2.imshow('img1 edge detection', img1_sob)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('img2', img2)
cv2.imshow('img2 edge detection', img2_sob)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('img3', img3)
cv2.imshow('img3 edge detection', img3_sob)
cv2.waitKey(0)
cv2.destroyAllWindows()
