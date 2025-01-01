import cv2
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['figure.autolayout'] = True  # 自動調整佈局防止標題和圖像重疊

def sobel(image):
    image = image / 255
    Gx = np.array([[-1, 0, 1] , [-2, 0, 2] , [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1] , [0, 0, 0] , [1, 2, 1]])
    sobel = np.zeros_like(image, dtype=float)

    # Perform convolution manually
    for c in range(image.shape[2]) :
        padded_image = np.pad(image[:,:,c], pad_width=1, mode='edge')
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i+3, j:j+3]
                sobel[i, j, c] = abs(np.sum(region * Gx)) + abs(np.sum(region * Gy))

    output = [image , sobel]
    titles = ["before sobel" , "after sobel"]

    # 創建 row x col 子圖佈局
    fig, axs = plt.subplots(1 , 2 , figsize = (10 ,  6))
    for ax, img, title in zip(axs , output , titles):
        ax.imshow(img)
        ax.set_title(title , fontsize = 12)
        ax.axis('off')  # 隱藏坐標軸

    # 調整子圖間距
    plt.tight_layout()
    plt.show()

    return

def gaussian_kernel(size, sigma=1.0):
    kernel = np.zeros((size, size), dtype=np.float32)
    offset = size // 2
    for x in range(-offset, offset + 1):
        for y in range(-offset, offset + 1):
            kernel[x + offset, y + offset] = (1 / ((2 * np.pi)**1/2 * sigma)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def convolve(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    result = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return result

def gradient(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Gx = convolve(image, Kx)
    Gy = convolve(image, Ky)

    Gx = Gx / np.max(Gx)
    Gy = Gy / np.max(Gy)

    print("Gx range:", Gx.min(), Gx.max())
    print("Gy range:", Gy.min(), Gy.max())

    magnitude = np.sqrt(Gx**2 + Gy**2)
    magnitude = magnitude / np.max(magnitude)
    direction = np.arctan2(Gy, Gx)

    print("magnitude range:", Gx.min(), Gx.max())

    return magnitude, direction

def non_maximum_suppression(magnitude, direction):
    h, w = magnitude.shape
    result = np.zeros((h, w), dtype=np.float32)
    direction = direction * (180.0 / np.pi)  # Convert to degrees
    direction = (direction + 180) % 180  # Normalize angles to [0, 180)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q, r = 255, 255
            if (0 <= direction[i, j] < 22.5) or (157.5 <= direction[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= direction[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= direction[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif 112.5 <= direction[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                result[i, j] = magnitude[i, j]

    return result

def double_threshold(image, low_ratio=0.05, high_ratio=0.15):
    high_threshold = image.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    strong = 255
    weak = 75
    result = np.zeros_like(image, dtype=np.uint8)
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result
    
    strong_edges = (image >= high_threshold)
    weak_edges = (image >= low_threshold) & (image < high_threshold)
    
    return strong_edges, weak_edges

def edge_tracking(image):
    h, w = image.shape
    strong = 255
    weak = 75
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if image[i, j] == weak:
                if (image[i-1:i+2, j-1:j+2] == strong).any():
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image

def canny(image):
    image = image / 255
    image = cv2.cvtColor(image , cv2.COLOR_RGB2GRAY)
    # Step 1: Gaussian Blur
    blurred = convolve(image, gaussian_kernel(5, sigma=1.4))

    # cv2.imshow("blur" , blurred)
    
    # Step 2: Gradient Calculation
    magnitude, direction = gradient(blurred)

    # cv2.imshow("mag" , magnitude)
    
    # Step 3: Non-Maximum Suppression
    thinned = non_maximum_suppression(magnitude, direction)

    # cv2.imshow("thin" , thinned)
    print("Thinned range:", thinned.min(), thinned.max())
    
    # Step 4: Double Threshold
    # strong_edges, weak_edges = double_threshold(thinned, low_ratio=0.05, high_ratio=0.15)
    double = double_threshold(thinned, low_ratio=0.05, high_ratio=0.15)

    # cv2.imshow("strong" , double)

    # cv2.imshow("strong" , strong_edges)
    # cv2.imshow("weak" , weak_edges)
    
    # Step 5: Edge Tracking by Hysteresis
    # edges = edge_tracking(strong_edges, weak_edges)
    edges = edge_tracking(double)

    output = [image , edges]
    titles = ["before canny" , "after canny"]

    # 創建 row x col 子圖佈局
    fig, axs = plt.subplots(1 , 2 , figsize = (10 ,  6))
    for ax, img, title in zip(axs , output , titles):
        ax.imshow(img , cmap='gray')
        ax.set_title(title , fontsize = 12)
        ax.axis('off')  # 隱藏坐標軸

    # 調整子圖間距
    plt.tight_layout()
    plt.show()
    
    return edges

# read image
dir = "./HW4_test_image/*.*"
for filepath in glob.glob(dir) :
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB).astype('float32')

    sobel(img)
    canny(img)