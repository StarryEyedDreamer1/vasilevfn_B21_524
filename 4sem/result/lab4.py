import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def rgb_to_gray(image):
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    

def sobel_operator(image):
    kernel_x = np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]])

    kernel_y = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
    
    Gx = np.abs(convolve2d(image, kernel_x, mode='same'))
    Gy = np.abs(convolve2d(image, kernel_y, mode='same'))
    G = Gx + Gy
    
    Gx_norm = (Gx - np.min(Gx)) / (np.max(Gx) - np.min(Gx)) * 255
    Gy_norm = (Gy - np.min(Gy)) / (np.max(Gy) - np.min(Gy)) * 255
    G_norm = (G - np.min(G)) / (np.max(G) - np.min(G)) * 255
    
    return Gx_norm, Gy_norm, G_norm

def binarize_image(image, threshold):
    binary_image = np.where(image > threshold, 255, 0)
    return binary_image.astype(np.uint8)

color_image = np.array(Image.open('Picture1.png'))

gray_image = rgb_to_gray(color_image).astype(np.uint8)

Gx, Gy, G = sobel_operator(gray_image)

threshold = 100
binary_image = binarize_image(G, threshold)

color_photo = Image.fromarray(color_image)
color_photo.show()
color_photo.save('color_photo.png')

gray_photo = Image.fromarray(gray_image, mode='L')
gray_photo.show()
gray_photo.save('gray_photo.png')

Gx_photo = Image.fromarray(Gx.astype(np.uint8), mode='L')
Gx_photo.show()
Gx_photo.save('Gx_photo.png')

Gy_photo = Image.fromarray(Gy.astype(np.uint8), mode='L')
Gy_photo.show()
Gy_photo.save('Gy_photo.png')

G_photo = Image.fromarray(G.astype(np.uint8), mode='L')
G_photo.show()
G_photo.save('G_photo.png')

binary_photo = Image.fromarray(binary_image, mode='L')
binary_photo.show()
binary_photo.save('binary_photo.png')

