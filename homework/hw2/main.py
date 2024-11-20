import cv2
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['figure.autolayout'] = True  # 自動調整佈局防止標題和圖像重疊

def Laplacian(f , filename) :
    # 傅立葉轉換
    F = np.fft.fftshift(np.fft.fft2(f))

    # 計算filter
    M , N = F.shape
    H = np.zeros((M , N) , dtype = np.float32)
    for u in range(M) :
        for v in range(N) :
            H[u , v] = -4 * np.pi * np.pi * ((u - M / 2) ** 2 + (v - N / 2) ** 2)

    # mask
    mask = H * F
    mask = np.real(np.fft.ifft2(np.fft.ifftshift(mask)))
    
    # 轉換到[-1 , 1]，因為mask出來的值可能遠大於f
    OldRange = np.max(mask) - np.min(mask)
    NewRange = 1 - -1
    maskScaled = (((mask - np.min(mask)) * NewRange) / OldRange) + -1

    # get g
    g = f - maskScaled
    g = np.clip(g , 0 , 1)

    output = [f , g]
    titles = ["before Laplacian" , "after Laplacian"]

    # 創建 1x2 子圖佈局
    fig, axs = plt.subplots(1 , 2, figsize = (8, 6))
    for ax, img, title in zip(axs , output , titles):
        ax.imshow(img , cmap = 'gray')
        ax.set_title(title , fontsize = 12)
        ax.axis('off')  # 隱藏坐標軸

    # 調整子圖間距
    plt.tight_layout()
    plt.show()

    return

def blurring(f) :
    F = np.fft.fftshift(np.fft.fft2(f))

    # Gaussian Low Pass Filter
    M , N = F.shape
    H = np.zeros((M , N), dtype=np.float32)
    D0 = 10
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            H[u , v] = np.exp(-D ** 2 / (2 * D0 * D0))

    # flp(x , y)
    FLP = H * F
    flp = np.abs(np.fft.ifft2(np.fft.ifftshift(FLP)))
    return flp

def unsharp(f , filename) :
    flp = blurring(f)
    gmask = f - flp
    ghp = f + gmask
    ghp = np.clip(ghp , 0 , 1)

    output = [f , ghp]
    titles = ["before unsharp" , "after unsharp"]

    # 創建 1x2 子圖佈局
    fig, axs = plt.subplots(1 , 2, figsize = (8, 6))
    for ax, img, title in zip(axs , output , titles):
        ax.imshow(img , cmap = 'gray')
        ax.set_title(title , fontsize = 12)
        ax.axis('off')  # 隱藏坐標軸

    # 調整子圖間距
    plt.tight_layout()
    plt.show()

    return 

def high_boost(f , filename) :
    flp = blurring(f)
    gmask = f - flp
    A = [1.7 , 2 , 2.7]
    output = [f]
    for AA in A :
        ghb = (AA - 1) * f + gmask
        ghb = np.clip(ghb , 0 , 1)
        output.append(ghb)

    titles = ["before high_boost" , f"A={A[0]}" , f"A={A[1]}" , f"A={A[2]}"]

    # 創建 2x2 子圖佈局
    fig, axs = plt.subplots(2 , 2, figsize = (15, 12))
    # 拉平讓它成為一維
    axs = axs.flatten()
    for ax, img, title in zip(axs , output , titles):
        ax.imshow(img , cmap = 'gray')
        ax.set_title(title , fontsize = 12)
        ax.axis('off')  # 隱藏坐標軸

    # 調整子圖間距
    plt.tight_layout()
    plt.show()

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