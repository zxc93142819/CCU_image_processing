import cv2
import os
import glob
import numpy as np
from matplotlib import pyplot as plt

def power_law(img , filename) :
    gamma_list = [2 , 0.5]
    c = 1
    h , w = img.shape

    output1 = np.zeros((h , w),dtype='uint8')
    output2 = np.zeros((h , w),dtype='uint8')
    output_list = [output1 , output2]

    for i in range(h) :
        for j in range(w) :
            for k in range(2) :
                t = 255 * c * pow((img[i][j] / 255) , gamma_list[k])
                if t > 255:
                    t = 255
                if t < 0:
                    t = 0
                output_list[k][i][j] = t

    # output1 = np.array(255*(img/255)**gamma1,dtype='uint8')
    # output2 = np.array(255*(img/255)**gamma2,dtype='uint8')



    # output ----------------------------------------------------------
    gamma_combine = cv2.hconcat([img , output_list[0] , output_list[1]])

    # 造一個空白區域來寫caption
    h , w = gamma_combine.shape
    blank_space = np.zeros((50, w , 1), dtype=np.uint8)

    # 文字設定
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1

    cv2.putText(blank_space, "origin", (int(w * 0.15), 30), font, font_scale, color, thickness)
    cv2.putText(blank_space, "gamma=" + str(gamma_list[0]), (int(w * 0.46), 30), font, font_scale, color, thickness)
    cv2.putText(blank_space, "gamma=" + str(gamma_list[1]), (int(w * 0.78), 30), font, font_scale, color, thickness)

    # 使用 vconcat 垂直拼在一起
    final_image = cv2.vconcat([gamma_combine, blank_space])

    cv2.imshow(f"compare of {filename} before and after gamma transform", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def histogram(img , filename) :
    hist_x = [i for i in range(256) ]
    h , w = img.shape

    # global and local histogram equalization
    grid_size =[[1 , 1] , [2 , 2] , [4 , 4] , [8 , 8]]
    output_img = [img]
    for k in range(4) :
        equalized_image = np.zeros_like(img)
        grid_h = h // grid_size[k][0]
        grid_w = w // grid_size[k][1]
        for i in range(grid_size[k][0]) :
            for j in range(grid_size[k][1]) :
                # 計算當前區域的邊界
                start_h = i * grid_h
                end_h = (i + 1) * grid_h if i != grid_size[k][0] - 1 else h
                start_w = j * grid_w
                end_w = (j + 1) * grid_w if j != grid_size[k][1] - 1 else w

                # 擷取當前區域的子圖像
                sub_image = img[start_h:end_h, start_w:end_w]
                
                # 計算sub_image之histogram
                origin_hist_y = [0] * 256
                for ii in range(grid_h) :
                    for jj in range(grid_w) :
                        origin_hist_y[sub_image[ii][jj]] += 1
                
                # 計算PDF and CDF
                pdf = [y / (grid_h * grid_w) for y in origin_hist_y]
                cdf = []
                cumulative_sum = 0
                for pdf_v in pdf:
                    cumulative_sum += pdf_v
                    cdf.append(cumulative_sum)

                # 將 CDF 映射到 [0, 255]，並四捨五入作為新的灰度值
                cdf_scaled = [int(cdf_v * 255) for cdf_v in cdf]
                
                # tranform it
                equalized_sub_image = np.zeros((grid_h , grid_w) , dtype='uint8')
                for ii in range(grid_h) :
                    for jj in range(grid_w) :
                        equalized_sub_image[ii][jj] = cdf_scaled[sub_image[ii][jj]]
                
                # 將均衡化後的區域放回結果圖像中
                equalized_image[start_h:end_h, start_w:end_w] = equalized_sub_image
        output_img.append(equalized_image)

    # calculate histogram
    output_histogram = []
    for output in output_img :
        h , w = output.shape
        hist_y = [0] * 256
        for i in range(h) :
            for j in range(w) :
                hist_y[output[i][j]] += 1
        output_histogram.append(hist_y)

    # output -------------------------------------------------
    histogram_combine = cv2.hconcat(output_img)

    # 造一個空白區域來寫caption
    h , w = histogram_combine.shape
    blank_space = np.zeros((50, w , 1), dtype=np.uint8)

    # 文字設定
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1

    cv2.putText(blank_space, "origin", (int(w * 0.08), 30), font, font_scale, color, thickness)
    cv2.putText(blank_space, "after global transform", (int(w * 0.22), 30), font, font_scale, color, thickness)
    cv2.putText(blank_space, "after local 2*2 transform", (int(w * 0.41), 30), font, font_scale, color, thickness)
    cv2.putText(blank_space, "after local 4*4 transform", (int(w * 0.62), 30), font, font_scale, color, thickness)
    cv2.putText(blank_space, "after local 8*8 transform", (int(w * 0.81), 30), font, font_scale, color, thickness)

    # 使用 vconcat 垂直拼在一起
    final_image = cv2.vconcat([histogram_combine, blank_space])

    cv2.imshow(f"compare of {filename} before and after histogram transform", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.bar(hist_x, output_histogram[0], color='b')
    plt.title(f"histogram of {filename} before transform")
    plt.xlabel("Gray level")
    plt.ylabel("frequency")
    plt.xlim(0,255)
    plt.show()

    plt.bar(hist_x, output_histogram[1], color='b')
    plt.title(f"histogram of {filename} after global transform")
    plt.xlabel("Gray level")
    plt.ylabel("frequency")
    plt.xlim(0,255)
    plt.show()

    plt.bar(hist_x, output_histogram[2], color='b')
    plt.title(f"histogram of {filename} after local 2*2 transform")
    plt.xlabel("Gray level")
    plt.ylabel("frequency")
    plt.xlim(0,255)
    plt.show()

    plt.bar(hist_x, output_histogram[3], color='b')
    plt.title(f"histogram of {filename} after local 4*4 transform")
    plt.xlabel("Gray level")
    plt.ylabel("frequency")
    plt.xlim(0,255)
    plt.show()

    plt.bar(hist_x, output_histogram[4], color='b')
    plt.title(f"histogram of {filename} after local 8*8 transform")
    plt.xlabel("Gray level")
    plt.ylabel("frequency")
    plt.xlim(0,255)
    plt.show()

def image_sharpening(img , filename) :
    mask1 = [0 , -1 , 0 , -1 , 5 , -1 , 0 , -1 , 0]
    mask2 = [-1 , -1 , -1 , -1 , 9 , -1 , -1 , -1 , -1]
    dir_x = [-1 , -1 , -1 , 0 , 0 , 0 , 1 , 1 , 1]
    dir_y = [-1 , 0 , 1 , -1 , 0 , 1 , -1 , 0 , 1]
    mask_list = [mask1 , mask2]
    h , w = img.shape

    output1 = np.zeros((h , w) , dtype = 'uint8')
    output2 = np.zeros((h , w) , dtype = 'uint8')
    output_list = [output1 , output2]
    for i in range(h) :
        for j in range(w) :
            for k in range(2) :
                # 邊緣不做 transform
                if(i == 0 or i == h - 1 or j == 0 or j == w - 1) :
                    output_list[k][i][j] = img[i][j]
                else :
                    t = 0
                    for index in range(9) :
                        t += mask_list[k][index] * img[i + dir_x[index]][j + dir_y[index]]
                    if(t > 255) :
                        t = 255
                    if(t < 0) :
                        t = 0
                    output_list[k][i][j] = t
    

    # output ----------------------------------------------------------
    laplacian_combine = cv2.hconcat([img , output_list[0] , output_list[1]])

    # 造一個空白區域來寫caption
    h , w = laplacian_combine.shape
    blank_space = np.zeros((50, w , 1), dtype=np.uint8)

    # 文字設定
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1

    cv2.putText(blank_space, "origin", (int(w * 0.15), 30), font, font_scale, color, thickness)
    cv2.putText(blank_space, "mask1" , (int(w * 0.46), 30), font, font_scale, color, thickness)
    cv2.putText(blank_space, "mask2" , (int(w * 0.78), 30), font, font_scale, color, thickness)

    # 使用 vconcat 垂直拼在一起
    final_image = cv2.vconcat([laplacian_combine, blank_space])

    cv2.imshow(f"compare of {filename} before and after laplacian transform", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# read image
dir = "./image/*.bmp"
# dir = "./HW1_test_image/*.bmp"
for filepath in glob.glob(dir) :
    img_gray = cv2.imread(filepath , cv2.IMREAD_GRAYSCALE)
    basename = os.path.basename(filepath)
    filename = os.path.splitext(basename)[0]

    power_law(img_gray , filename)
    histogram(img_gray , filename)
    image_sharpening(img_gray , filename)

    # bmp_image.close()