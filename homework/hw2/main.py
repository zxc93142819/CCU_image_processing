import cv2
import os
import glob
import numpy as np
from matplotlib import pyplot as plt

def Laplacian(f , filename) :
    # 傅立葉轉換
    F = np.fft.fftshift(np.fft.fft2(f))

    # 計算filter
    U , V = F.shape
    H = np.zeros((U , V) , dtype = np.float32)
    for u in range(U) :
        for v in range(V) :
            H[u , v] = -4 * np.pi * np.pi * ((u - U / 2) ** 2 + (v - V / 2) ** 2)

    # mask
    mask = H * F
    mask = np.real(np.fft.ifft2(np.fft.ifftshift(mask)))
    
    # 轉換到[-1 , 1]，因為mask出來的值可能遠大於f
    OldRange = np.max(mask) - np.min(mask)
    NewRange = 1 - -1
    maskScaled = (((mask - np.min(mask)) * NewRange) / OldRange) + -1

    # output
    c = -1
    g = f + c * maskScaled
    g = np.clip(g , 0 , 1)

    cv2.imshow(filename , g)
    cv2.waitKey()

    return

def unsharp(f , filename) :
    F = np.fft.fftshift(np.fft.fft2(f))
    return 

def high_boost(f , filename) :
    F = np.fft.fftshift(np.fft.fft2(f))
    return

def homomorphic(f , filename) :
    F = np.fft.fftshift(np.fft.fft2(f))
    return

# read image
dir = "./HW2_test_image/*.*"
for filepath in glob.glob(dir) :
    img = cv2.imread(filepath , cv2.IMREAD_GRAYSCALE)
    basename = os.path.basename(filepath)
    filename = os.path.splitext(basename)[0]
    f = img / 255

    Laplacian(f , filename)
    unsharp(f , filename)
    high_boost(f , filename)
    homomorphic(f , filename)