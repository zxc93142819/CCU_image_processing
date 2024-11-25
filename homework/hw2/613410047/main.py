import cv2
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['figure.autolayout'] = True  # 自動調整佈局防止標題和圖像重疊

def output_image(row , col , h , w , output , titles) :
    # 創建 row x col 子圖佈局
    fig, axs = plt.subplots(row , col , figsize = (h ,  w))
    if(row >= 2) :
        axs = axs.flatten()
    for ax, img, title in zip(axs , output , titles):
        ax.imshow(img , cmap = 'gray')
        ax.set_title(title , fontsize = 12)
        ax.axis('off')  # 隱藏坐標軸

    # 調整子圖間距
    plt.tight_layout()
    plt.show()

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

    output_image(row = 1 , col = 2 , h = 8 , w = 6 , output = output , titles = titles)

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

    output_image(row = 1 , col = 2 , h = 8 , w = 6 , output = output , titles = titles)

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

    output_image(row = 1 , col = 4 , h = 15 , w = 8 , output = output , titles = titles)

    return

def homomorphic(f , filename) :
    # 取ln
    z = np.log1p(np.array(f) , dtype = "float")
    
    # 傅立葉轉換
    Z = np.fft.fftshift(np.fft.fft2(z))

    # gaussian filter
    M , N = f.shape
    D0 = 80
    c = 2
    rL = 0.5
    rH = 2
    H = np.zeros((M , N) , dtype = np.float32)
    for u in range(M) :
        for v in range(N) :
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            H[u , v] = (rH - rL) * (1.0 - np.exp( -c * (D ** 2 / D0 ** 2) )) + rL
    
    # S(u,v) = H(u,v)Z(u,v)
    S = H * Z

    # s = S的反傅立葉轉換
    s = np.real(np.fft.ifft2(np.fft.ifftshift(S)))
    # s = np.fft.ifft2(np.fft.ifftshift(S))

    # g = exp[s(x,y)]
    g = np.exp(s)
    # g = np.abs(np.exp(s))
    # g = np.clip(g , 0 , 1)

    output = [f , g]
    titles = ["before homomorphic" , "after homomorphic"]

    output_image(row = 1 , col = 2 , h = 8 , w = 6 , output = output , titles = titles)

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