import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
    
    return kernel / np.sum(kernel)

def convolve(image, kernel):
    image_height, image_width = image.shape
    kernel_size = kernel.shape[0]
    padding = kernel_size // 2
    
    padded_image = np.pad(image, padding, mode='reflect')
    output = np.zeros_like(image)
    
    for i in range(image_height):
        for j in range(image_width):
            output[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)
    
    return output

def compute_gradient(image):
    # Sobel operators
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    gradient_x = convolve(image, sobel_x)
    gradient_y = convolve(image, sobel_y)
    
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    
    return gradient_magnitude, gradient_direction

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    height, width = gradient_magnitude.shape
    output = np.zeros_like(gradient_magnitude)
    
    # Convert angles to degrees and shift to positive values
    angle = gradient_direction * 180 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            q = 255
            r = 255
            
            # Angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = gradient_magnitude[i, j+1]
                r = gradient_magnitude[i, j-1]
            # Angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = gradient_magnitude[i+1, j-1]
                r = gradient_magnitude[i-1, j+1]
            # Angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = gradient_magnitude[i+1, j]
                r = gradient_magnitude[i-1, j]
            # Angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = gradient_magnitude[i-1, j-1]
                r = gradient_magnitude[i+1, j+1]
            
            if (gradient_magnitude[i,j] >= q) and (gradient_magnitude[i,j] >= r):
                output[i,j] = gradient_magnitude[i,j]
            else:
                output[i,j] = 0
                
    return output

def double_threshold(image, low_ratio=0.05, high_ratio=0.15):
    high_threshold = image.max() * high_ratio
    low_threshold = high_threshold * low_ratio
    
    strong_edges = (image >= high_threshold)
    weak_edges = (image >= low_threshold) & (image < high_threshold)
    
    return strong_edges, weak_edges

def hysteresis(strong_edges, weak_edges):
    height, width = strong_edges.shape
    output = np.copy(strong_edges)
    
    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, -1, 1, -1, 0, 1]
    
    # Connect weak edges to strong edges
    changed = True
    while changed:
        changed = False
        for i in range(1, height-1):
            for j in range(1, width-1):
                if weak_edges[i, j] and not output[i, j]:
                    # Check if any neighboring pixel is a strong edge
                    for k in range(8):
                        if output[i + dx[k], j + dy[k]]:
                            output[i, j] = True
                            changed = True
                            break
    
    return output

def canny_edge_detection(image, gaussian_size=5, gaussian_sigma=1.4):
    # 1. Apply Gaussian filter
    gaussian_filter = gaussian_kernel(gaussian_size, gaussian_sigma)
    smoothed = convolve(image, gaussian_filter)

    plt.imshow(smoothed, cmap='gray')
    
    # 2. Compute gradients
    gradient_magnitude, gradient_direction = compute_gradient(smoothed)
    
    plt.imshow(gradient_magnitude, cmap='gray')

    # 3. Apply non-maximum suppression
    suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)
    
    # 4. Double threshold
    strong_edges, weak_edges = double_threshold(suppressed)
    
    # 5. Edge tracking by hysteresis
    final_edges = hysteresis(strong_edges, weak_edges)
    
    return final_edges.astype(np.uint8) * 255

# Example usage:
if __name__ == "__main__":
    # 讀取圖片
    image = np.array(Image.open('./HW4_test_image/image1.jpg').convert('L'))
    edges = canny_edge_detection(image)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')
    plt.axis('off')
    plt.show()