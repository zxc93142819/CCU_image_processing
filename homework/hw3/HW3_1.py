import cv2
import numpy as np
from matplotlib import pyplot as plt

aloe = cv2.imread('aloe.jpg')
church = cv2.imread('church.jpg')
house = cv2.imread('house.jpg')
kitchen = cv2.imread('kitchen.jpg')

(row1, col1, color1) = aloe.shape[0], aloe.shape[1], aloe.shape[2]
(row2, col2, color2) = church.shape[0], church.shape[1], church.shape[2]
(row3, col3, color3) = house.shape[0], house.shape[1], house.shape[2]
(row4, col4, color4) = kitchen.shape[0], kitchen.shape[1], kitchen.shape[2]

neighbor = np.array([[-1,-1], [0,-1],  [1,-1],
                     [-1,0],  [0,0],   [1,0],
                     [-1,1],  [0,1],   [1,1]])

laplacian = np.array([ 0,-1, 0,
                      -1, 5,-1,
                       0,-1, 0])

aloe_laplacian = np.zeros((row1, col1, color1), dtype = 'uint8')
for i in range(1, row1-1):
    for j in range(1, col1-1):
        for k in range(0, 3):
            sum = 0
            for m in range(0, 9):
                r = i + neighbor[m][0]
                c = j + neighbor[m][1]
                sum = sum + aloe[r][c][k] * laplacian[m]
            if sum > 255:
                sum = 255
            if sum < 0:
                sum = 0
            aloe_laplacian[i][j][k] = sum

church_laplacian = np.zeros((row2, col2, color2), dtype = 'uint8')
for i in range(1, row2-1):
    for j in range(1, col2-1):
        for k in range(0, 3):
            sum = 0
            for m in range(0, 9):
                r = i + neighbor[m][0]
                c = j + neighbor[m][1]
                sum = sum + church[r][c][k] * laplacian[m]
            if sum > 255:
                sum = 255
            if sum < 0:
                sum = 0
            church_laplacian[i][j][k] = sum

house_laplacian = np.zeros((row3, col3, color3), dtype = 'uint8')
for i in range(1, row3-1):
    for j in range(1, col3-1):
        for k in range(0, 3):
            sum = 0
            for m in range(0, 9):
                r = i + neighbor[m][0]
                c = j + neighbor[m][1]
                sum = sum + house[r][c][k] * laplacian[m]
            if sum > 255:
                sum = 255
            if sum < 0:
                sum = 0
            house_laplacian[i][j][k] = sum

kitchen_laplacian = np.zeros((row4, col4, color4), dtype = 'uint8')
for i in range(1, row4-1):
    for j in range(1, col4-1):
        for k in range(0, 3):
            sum = 0
            for m in range(0, 9):
                r = i + neighbor[m][0]
                c = j + neighbor[m][1]
                sum = sum + kitchen[r][c][k] * laplacian[m]
            if sum > 255:
                sum = 255
            if sum < 0:
                sum = 0
            kitchen_laplacian[i][j][k] = sum

cv2.imshow('Aloe', aloe)
cv2.imshow('Aloe after Enhancement', aloe_laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Church', church)
cv2.imshow('Church after Enhancement', church_laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('House', house)
cv2.imshow('House after Enhancement', house_laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Kitchen', kitchen)
cv2.imshow('Kitchen after Enhancement', kitchen_laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()