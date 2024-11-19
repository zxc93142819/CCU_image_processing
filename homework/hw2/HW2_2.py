import cv2
import numpy as np
from matplotlib import pyplot as plt

moon = cv2.imread('HW2_test_image/blurry_moon.tif')
skeleton = cv2.imread('HW2_test_image/skeleton_orig.bmp')

(row1, col1) = moon.shape[0], moon.shape[1]
moon_gray = np.zeros((row1, col1), dtype = 'float32')
for i in range(0, row1):
    for j in range(0, col1):
        moon_gray[i][j] = moon[i][j][0]

(row2, col2) = skeleton.shape[0], skeleton.shape[1]
skeleton_gray = np.zeros((row2, col2))
for i in range(0, row2):
    for j in range(0, col2):
        skeleton_gray[i][j] = skeleton[i][j][0]

# 正規化
moon_gray = moon_gray / 255
skeleton_gray = skeleton_gray / 255
# 快速傅立葉變換演演算法得到頻率分佈，將空間域轉化為頻率域
moon_f = np.fft.fft2(moon_gray)
skeleton_f = np.fft.fft2(skeleton_gray)
# 預設結果中心點位置是在左上角,通過下述程式碼將中心點轉移到中間位置，將低頻部分移動到影象中心
moon_f = np.fft.fftshift(moon_f)
skeleton_f = np.fft.fftshift(skeleton_f)

#Laplacian operator
moon_h = np.zeros((row1, col1), dtype = 'float32')
skeleton_h = np.zeros((row2, col2), dtype = 'float32')
for i in range(row1):
    for j in range(col1):
        moon_h[i,j] = -4*np.pi*np.pi*((i-row1/2)**2 + (j-col1/2)**2)

for i in range(row2):
    for j in range(col2):
        skeleton_h[i,j] = -4*np.pi*np.pi*((i-row2/2)**2 + (j-col2/2)**2)

moon_lap = moon_h * moon_f
skeleton_lap = skeleton_h * skeleton_f

# 將頻譜影象的中心低頻部分移動至左上角
moon_lap = np.fft.ifftshift(moon_lap)
skeleton_lap = np.fft.ifftshift(skeleton_lap)
# 影象逆傅立葉變換，返回一個複數陣列
moon_lap = np.fft.ifft2(moon_lap)
moon_lap = np.real(moon_lap)

skeleton_lap = np.fft.ifft2(skeleton_lap)
skeleton_lap = np.real(skeleton_lap)

#轉換Laplacian影像的值到區間[-1,1]
new_range = 1 - (-1)
old_range = np.max(moon_lap) - np.min(moon_lap)
moon_lapScaled = (((moon_lap - np.min(moon_lap)) * new_range) / old_range) + (-1)
moon_laplacian = moon_gray - moon_lapScaled
moon_laplacian = np.clip(moon_laplacian, 0, 1)

old_range = np.max(skeleton_lap) - np.min(skeleton_lap)
skeleton_lapScaled = (((skeleton_lap - np.min(skeleton_lap)) * new_range) / old_range) + (-1)
skeleton_laplacian = skeleton_gray - skeleton_lapScaled
skeleton_laplacian = np.clip(skeleton_laplacian, 0, 1)

#unsharp masking
moon_h = np.zeros((row1, col1), dtype = 'float32')
skeleton_h = np.zeros((row2, col2), dtype = 'float32')
D0 = 10
for i in range(row1):
    for j in range(col1):
        D = np.sqrt((i-row1/2)**2 + (j-col1/2)**2)
        moon_h[i,j] = np.exp(-D**2/(2*D0*D0))

for i in range(row2):
    for j in range(col2):
        D = np.sqrt((i-row2/2)**2 + (j-col2/2)**2)
        skeleton_h[i,j] = np.exp(-D**2/(2*D0*D0))

# creat fLP(x,y) (smoothed image)
moon_fLP = moon_h * moon_f
moon_fLP = np.fft.ifftshift(moon_fLP)

skeleton_fLP = skeleton_h * skeleton_f
skeleton_fLP = np.fft.ifftshift(skeleton_fLP)

# 影象逆傅立葉變換，返回一個複數陣列
moon_fLP = np.fft.ifft2(moon_fLP)
moon_fLP = np.abs(moon_fLP)

skeleton_fLP = np.fft.ifft2(skeleton_fLP)
skeleton_fLP = np.abs(skeleton_fLP)

# create mask & unsharp masking
moon_mask = moon_gray - moon_fLP
moon_unsharp = moon_gray + moon_mask
moon_unsharp = np.clip(moon_unsharp, 0, 1)

skeleton_mask = skeleton_gray - skeleton_fLP
skeleton_unsharp = skeleton_gray + skeleton_mask
skeleton_unsharp = np.clip(skeleton_unsharp, 0, 1)

#high-boost filtering
A = 1.7
moon_hbp = (A-1) * moon_gray + moon_mask
moon_hbp = np.clip(moon_hbp, 0, 1)

skeleton_hbp = (A-1) * skeleton_gray + skeleton_mask
skeleton_hbp = np.clip(skeleton_hbp, 0, 1)

cv2.imshow("幹" , skeleton_laplacian)
cv2.waitKey()

plt.subplot(121), plt.imshow(moon, "gray"), plt.title('origin')
plt.axis('off')
plt.subplot(122), plt.imshow(moon_laplacian, "gray"), plt.title('Laplacian')
plt.axis('off')
plt.show()

plt.subplot(121), plt.imshow(moon, "gray"), plt.title('origin')
plt.axis('off')
plt.subplot(122), plt.imshow(moon_unsharp, "gray"), plt.title('unsharp masking')
plt.axis('off')
plt.show()

plt.subplot(121), plt.imshow(moon, "gray"), plt.title('origin')
plt.axis('off')
plt.subplot(122), plt.imshow(moon_hbp, "gray"), plt.title('high-boost filtering')
plt.axis('off')
plt.show()

plt.subplot(121), plt.imshow(skeleton, "gray"), plt.title('origin')
plt.axis('off')
plt.subplot(122), plt.imshow(skeleton_laplacian, "gray"), plt.title('Laplacian')
plt.axis('off')
plt.show()

plt.subplot(121), plt.imshow(skeleton, "gray"), plt.title('origin')
plt.axis('off')
plt.subplot(122), plt.imshow(skeleton_unsharp, "gray"), plt.title('unsharp masking')
plt.axis('off')
plt.show()

plt.subplot(121), plt.imshow(skeleton, "gray"), plt.title('origin')
plt.axis('off')
plt.subplot(122), plt.imshow(skeleton_hbp, "gray"), plt.title('high-boost filtering')
plt.axis('off')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()