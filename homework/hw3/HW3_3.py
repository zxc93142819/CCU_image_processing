import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

def liner(img):
    row, col, color = img.shape[0], img.shape[1], img.shape[2]
    img = img / 255.0
    lin_img = np.zeros((row, col, color), dtype = 'float32')
    for i in range(0, row):
        for j in range(0, col):
            for k in range(0, color):
                if img[i][j][k] <= 0.04045:
                    lin_img[i][j][k] = img[i][j][k] / 12.92
                else:
                    lin_img[i][j][k] = pow((img[i][j][k] + 0.055) / 1.055, 2.4)
    return lin_img

def RGB2XYZ(img):
    lin_img = liner(img)
    matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
    xyz_img = np.dot(lin_img, matrix.T)
    return xyz_img

def h_fun(q):
    if q > 0.008856:
        return math.pow(q, 1/3)
    else:
        return ((7.787 * q) + (16/116))

def XYZ2Lab(img):
    row, col, color = img.shape[0], img.shape[1], img.shape[2]
    x, y, z = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    xw = 0.950456
    yw = 1.0
    zw = 1.088754
    lab_img = np.zeros((row, col, color))
    for i in range(0, row):
        for j in range(0, col):
            lab_img[i][j][0] = (116 * h_fun(y[i][j] / yw)) - 16 #L
            lab_img[i][j][1] = 500 * (h_fun(x[i][j] / xw) - h_fun(y[i][j] / yw)) #a
            lab_img[i][j][2] = 200 * (h_fun(y[i][j] / yw) - h_fun(z[i][j] / zw)) #b
    return lab_img

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
                sum = sum + img[r][c][0] * laplacian[m]
            if sum > 255:
                sum = 255
            if sum < 0:
                sum = 0
            img_lap[i][j][0] = sum
            img_lap[i][j][1] = img[r][c][1]
            img_lap[i][j][2] = img[r][c][2]
    return img_lap

def h_inv(q):
    if q > (6/29):
        return pow(q, 3)
    else:
        return ((q - 16 / 116) * 3 * pow(6/29, 2))

def Lab2XYZ(img):
    row, col, color = img.shape[0], img.shape[1], img.shape[2]
    L, a, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    xw = 0.950456
    yw = 1.0
    zw = 1.088754
    xyz_img = np.zeros((row, col, color), dtype = 'float32')
    for i in range(0, row):
        for j in range(0, col):
            xyz_img[i][j][1] = (L[i][j] + 16) / 116.0
            xyz_img[i][j][0] = xyz_img[i][j][1] + a[i][j] / 500.0
            xyz_img[i][j][2] = xyz_img[i][j][1] - b[i][j] / 200.0

            xyz_img[i][j][1] = h_inv(xyz_img[i][j][1]) * yw
            xyz_img[i][j][0] = h_inv(xyz_img[i][j][0]) * xw
            xyz_img[i][j][2] = h_inv(xyz_img[i][j][2]) * zw
    return xyz_img

def non_liner(img):
    row, col, color = img.shape[0], img.shape[1], img.shape[2]
    non_img = np.zeros((row, col, color))
    for i in range(0, row):
        for j in range(0, col):
            for k in range(0, color):
                if img[i][j][k] > 0.0031308:
                    non_img[i][j][k] = 1.055 * pow(img[i][j][k], 1 / 2.4) - 0.055
                else:
                    non_img[i][j][k] = 12.92 * img[i][j][k]
    return non_img

def XYZ2RGB(img):
    matrix = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                       [-0.9692660,  1.8760108,  0.0415560],
                       [ 0.0556434, -0.2040259,  1.0572252]])
    rgb_img = np.dot(img, matrix.T)
    non_img = non_liner(rgb_img)
    non_img = np.array(np.clip(non_img, 0, 1) * 255, dtype = 'uint8')
    return non_img

aloe = cv2.imread('./HW3_test_image/aloe.jpg')
church = cv2.imread('./HW3_test_image/church.jpg')
house = cv2.imread('./HW3_test_image/house.jpg')
kitchen = cv2.imread('./HW3_test_image/kitchen.jpg')

aloe = cv2.cvtColor(aloe , cv2.COLOR_BGR2RGB)

aloe_xyz = RGB2XYZ(aloe)
aloe_lab = XYZ2Lab(aloe_xyz)
print(aloe_lab)
print("-----------------------------")
aloe_lap = Lap(aloe_lab)
print(aloe_lap)
# aloe_XYZ = Lab2XYZ(aloe_lap)
# aloe_rgb = XYZ2RGB(aloe_XYZ)