import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

def HSI(img):
    row, col, color = img.shape[0], img.shape[1], img.shape[2]
    img = img / 255.0
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    hsi_img = np.zeros((row, col, color))
    for i in range(0, row):
        for j in range(0, col):
            n1 = 0.5 * ((r[i][j] - g[i][j]) + (r[i][j] - b[i][j]))
            n2 = math.sqrt(((r[i][j] - g[i][j]) ** 2) + (r[i][j] - b[i][j]) * (g[i][j] - b[i][j]))
            if n1 == 0 and n2 == 0:
                theta = math.pi / 2
            else:
                theta = math.acos(n1 / n2)
            theta = (180 / math.pi) * theta
            if b[i][j] <= g[i][j]:
                H = theta
            else:
                H = 360 - theta

            min_rgb = min(min(b[i][j], g[i][j]), r[i][j])
            sum = b[i][j] + g[i][j] + r[i][j]
            if sum == 0:
                S = 0
            else:
                S = 1 - (3 * min_rgb / (sum + 1e-6))

            I = sum / 3.0
            hsi_img[i][j][0] = H
            hsi_img[i][j][1] = S
            hsi_img[i][j][2] = I
    return hsi_img

def Lap(img):
    neighbor = np.array([[-1, -1], [0, -1], [1, -1],
                         [-1, 0], [0, 0], [1, 0],
                         [-1, 1], [0, 1], [1, 1]])

    laplacian = np.array([0, -1, 0,
                          -1, 5, -1,
                          0, -1, 0])
    row, col, color = img.shape[0], img.shape[1], img.shape[2]
    img_lap = np.zeros((row, col, color))
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            sum = 0
            for m in range(0, 9):
                r = i + neighbor[m][0]
                c = j + neighbor[m][1]
                sum = sum + img[r][c][2] * laplacian[m]
            if sum > 255:
                sum = 255
            if sum < 0:
                sum = 0
            img_lap[i][j][0] = img[r][c][0]
            img_lap[i][j][1] = img[r][c][1]
            img_lap[i][j][2] = sum
    return img_lap

def RGB(img):
    row, col, color = img.shape[0], img.shape[1], img.shape[2]
    rgb_img = np.zeros((row, col, color))
    H, S, I = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    for i in range(0, row):
        for j in range(0, col):
            if S[i][j] < 1e-6:
                R = I[i][j]
                G = I[i][j]
                B = I[i][j]
            else:
                if (H[i][j] >= 0) and (H[i][j] < 120):
                    H[i][j] = (math.pi / 180) * H[i][j]
                    B = I[i][j] * (1 - S[i][j])
                    R = I[i][j] * (1 + ((S[i][j]* math.cos(H[i][j]))/ math.cos((math.pi / 180) * 60 - H[i][j])))
                    G = 3 * I[i][j] - (B + R)
                elif (H[i][j] >= 120) & (H[i][j] < 240):
                    H[i][j] = H[i][j] - 120
                    H[i][j] = (math.pi / 180) * H[i][j]
                    R = I[i][j] * (1 - S[i][j])
                    G = I[i][j] * (1 + ((S[i][j] * math.cos(H[i][j])) / math.cos((math.pi / 180) * 60 - H[i][j])))
                    B = 3 * I[i][j] - (R + G)
                else:
                    H[i][j] = H[i][j] - 240
                    H[i][j] = (math.pi / 180) * H[i][j]
                    G = I[i][j] * (1 - S[i][j])
                    B = I[i][j] * (1 + ((S[i][j] * math.cos(H[i][j])) / math.cos((math.pi / 180) * 60 - H[i][j])))
                    R = 3 * I[i][j] - (G + B)
            rgb_img[i][j][0] = B
            rgb_img[i][j][1] = G
            rgb_img[i][j][2] = R
    rgb_img = rgb_img.astype('float32')
    return rgb_img

aloe = cv2.imread('./HW3_test_image/aloe.jpg')
church = cv2.imread('./HW3_test_image/church.jpg')
house = cv2.imread('./HW3_test_image/house.jpg')
kitchen = cv2.imread('./HW3_test_image/kitchen.jpg')

aloe_hsi = HSI(aloe)
aloe_lap = Lap(aloe_hsi)
aloe_rgb = RGB(aloe_lap)

church_hsi = HSI(church)
church_lap = Lap(church_hsi)
church_rgb = RGB(church_lap)

house_hsi = HSI(house)
house_lap = Lap(house_hsi)
house_rgb = RGB(house_lap)

kitchen_hsi = HSI(kitchen)
kitchen_lap = Lap(kitchen_hsi)
kitchen_rgb = RGB(kitchen_lap)

cv2.imshow('Aloe', aloe)
cv2.imshow('Aloe after Enhancement', aloe_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Church', church)
cv2.imshow('Church after Enhancement', church_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('House', house)
cv2.imshow('House after Enhancement', house_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Kitchen', kitchen)
cv2.imshow('Kitchen after Enhancement', kitchen_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()