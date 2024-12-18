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
        ax.imshow(img)
        ax.set_title(title , fontsize = 12)
        ax.axis('off')  # 隱藏坐標軸

    # 調整子圖間距
    plt.tight_layout()
    plt.show()

def apply_laplacian(image):
    kernel = np.array([[0 , -1 , 0] , [-1 , 5 , -1] , [0 , -1 , 0]])
    # Pad the image to handle border pixels
    padded_image = np.pad(image, pad_width=1, mode='edge')
    result = np.zeros_like(image, dtype=float)

    # Perform convolution manually
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+3, j:j+3]
            result[i, j] = np.sum(region * kernel)

    # Clip values to maintain valid range
    # result = np.clip(result, 0, 1)
    return result

def RGB_to_HSI(rgb) :
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    intensity = (r + g + b) / 3
    minimum = np.minimum(np.minimum(r, g), b)

    # Add epsilon to avoid divide by zero
    saturation = 1 - 3 * minimum / (r + g + b + 1e-6)

    numerator = 0.5 * ((r - g) + (r - b))
    # Add epsilon to avoid divide by zero
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-6
    theta = np.arccos(numerator / denominator)

    hue = np.where(b <= g, theta, 2 * np.pi - theta)
    hue = hue / (2 * np.pi)  # Normalize hue to [0, 1]

    hsi_image = np.stack((hue, saturation, intensity), axis=-1)
    return hsi_image

def HSI_to_RGB(hsi) :
    h, s, i = hsi[:,:,0], hsi[:,:,1], hsi[:,:,2]
    h = h * 2 * np.pi  # from [0, 1] convert to radians

    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)

    # RGB computation based on HSI ranges
    RG_sector = (0 <= h) & (h < 2 * np.pi / 3)
    b[RG_sector] = i[RG_sector] * (1 - s[RG_sector])
    r[RG_sector] = i[RG_sector] * (1 + s[RG_sector] * np.cos(h[RG_sector]) / np.cos(np.pi / 3 - h[RG_sector]))
    g[RG_sector] = 3 * i[RG_sector] - (r[RG_sector] + b[RG_sector])

    GB_sector = (2 * np.pi / 3 <= h) & (h < 4 * np.pi / 3)
    h[GB_sector] -= 2 * np.pi / 3
    r[GB_sector] = i[GB_sector] * (1 - s[GB_sector])
    g[GB_sector] = i[GB_sector] * (1 + s[GB_sector] * np.cos(h[GB_sector]) / np.cos(np.pi / 3 - h[GB_sector]))
    b[GB_sector] = 3 * i[GB_sector] - (r[GB_sector] + g[GB_sector])

    BR_sector = (4 * np.pi / 3 <= h) & (h <= 2 * np.pi)
    h[BR_sector] -= 4 * np.pi / 3
    g[BR_sector] = i[BR_sector] * (1 - s[BR_sector])
    b[BR_sector] = i[BR_sector] * (1 + s[BR_sector] * np.cos(h[BR_sector]) / np.cos(np.pi / 3 - h[BR_sector]))
    r[BR_sector] = 3 * i[BR_sector] - (g[BR_sector] + b[BR_sector])

    rgb_image = np.stack((r, g, b), axis=-1)
    rgb_image = np.clip(rgb_image, 0, 1)
    return rgb_image

def RGB_to_Lab(rgb) :
    # Convert to XYZ
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    r = np.where(r > 0.04045, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
    g = np.where(g > 0.04045, ((g + 0.055) / 1.055) ** 2.4, g / 12.92)
    b = np.where(b > 0.04045, ((b + 0.055) / 1.055) ** 2.4, b / 12.92)

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Reference white D65，使 RGB 與 L*a*b* 等範圍映射
    x, y, z = x / 0.950456, y / 1.00000, z / 1.088754

    x = np.where(x > 0.008856, x ** (1/3), (7.787 * x) + (16 / 116))
    y = np.where(y > 0.008856, y ** (1/3), (7.787 * y) + (16 / 116))
    z = np.where(z > 0.008856, z ** (1/3), (7.787 * z) + (16 / 116))

    l = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    lab_image = np.stack((l, a, b), axis=-1)
    return lab_image

def Lab_to_RGB(lab) :
    l, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]

    y = (l + 16) / 116
    x = a / 500 + y
    z = y - b / 200
    
    x = np.where(x ** 3 > 0.008856, x ** 3, (x - 16 / 116) / 7.787)
    y = np.where(y ** 3 > 0.008856, y ** 3, (y - 16 / 116) / 7.787)
    z = np.where(z ** 3 > 0.008856, z ** 3, (z - 16 / 116) / 7.787)

    # Reference white D65，使 RGB 與 L*a*b* 等範圍映射
    x, y, z = x * 0.950456, y * 1.00000, z * 1.088754

    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    r = np.where(r > 0.0031308, 1.055 * (np.sign(r) * (np.abs(r)) ** (1 / 2.4)) - 0.055, r * 12.92)
    g = np.where(g > 0.0031308, 1.055 * (np.sign(g) * (np.abs(g)) ** (1 / 2.4)) - 0.055, g * 12.92)
    b = np.where(b > 0.0031308, 1.055 * (np.sign(b) * (np.abs(b)) ** (1 / 2.4)) - 0.055, b * 12.92)

    rgb_image = np.stack((r, g, b), axis=-1)
    rgb_image = np.clip(rgb_image, 0, 1)
    return rgb_image

def RGB(img) :
    result = np.zeros_like(img, dtype=float)
    for c in range(3) :
        result[:,:,c] = apply_laplacian(img[:,:,c])
        result[:,:,c] = np.clip(result[:,:,c], 0, 1)

    output = [img , result]
    titles = ["RGB before" , "RGB after"]
    output_image(row = 1 , col = 2 , h = 10 , w = 6 , output = output , titles = titles)

    return

def HSI(img) :
    hsi = RGB_to_HSI(img)
    lap = np.zeros_like(hsi, dtype=float)
    lap[:,:,0] = hsi[:,:,0]
    lap[:,:,1] = hsi[:,:,1]
    lap[:,:,2] = apply_laplacian(hsi[:,:,2])
    lap[:,:,2] = np.clip(lap[:,:,2], 0, 1)
    result = HSI_to_RGB(lap)

    output = [img , result]
    titles = ["HSI before" , "HSI after"]
    output_image(row = 1 , col = 2 , h = 10 , w = 6 , output = output , titles = titles)
    
    return

def Lab(img) :
    lab = RGB_to_Lab(img)
    lap = np.zeros_like(lab, dtype=float)
    lap[:,:,1] = lab[:,:,1]
    lap[:,:,2] = lab[:,:,2]
    lap[:,:,0] = apply_laplacian(lab[:,:,0])
    lap[:,:,0] = np.clip(lap[:,:,0], 0, 100)
    result = Lab_to_RGB(lap)

    output = [img , result]
    titles = ["L*a*b* before" , "L*a*b* after"]
    output_image(row = 1 , col = 2 , h = 10 , w = 6 , output = output , titles = titles)

    return

# read image
dir = "./HW3_test_image/*.*"
for filepath in glob.glob(dir) :
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    img = img/255
    RGB(img)
    HSI(img)
    Lab(img)