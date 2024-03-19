from PIL import Image
import numpy as np


def color_to_gray_weighted(input_image_path, output_image_path):
    input_image = Image.open(input_image_path)
    input_array = np.array(input_image)

    gray_array = np.dot(input_array[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    gray_image = Image.fromarray(gray_array)

    gray_image.save(output_image_path, 'BMP')

def sauvola_binarization(input_image_path, output_image_path, window_size=15, k=0.2, r=128):

    gray_image = Image.open(input_image_path)
    gray_array = np.array(gray_image)

    binarized_array = np.zeros_like(gray_array)

    for i in range(0, gray_array.shape[0] - window_size):
        for j in range(0, gray_array.shape[1] - window_size):
            window = gray_array[i:i+window_size, j:j+window_size]
            mean = np.mean(window)
            std = np.std(window)
            threshold = mean * (1 + k * (std / r - 1))
            binarized_array[i:i+window_size, j:j+window_size] = np.where(window < threshold, 0, 255)

    binarized_image = Image.fromarray(binarized_array)

    binarized_image.save(output_image_path, 'BMP')

# Пример использования функций
color_to_gray_weighted("im1.png", "output_grayscale_image.bmp")
sauvola_binarization("output_grayscale_image.bmp", "output_binarized_image.bmp")


