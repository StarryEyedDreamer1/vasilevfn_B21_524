from PIL import Image
import numpy as np


def color_to_gray_weighted(input_image_path, output_image_path):
    input_image = Image.open(input_image_path)
    input_array = np.array(input_image)

    gray_array = np.dot(input_array[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    gray_image = Image.fromarray(gray_array)

    gray_image.save(output_image_path, 'BMP')


# ������ ������������� �������
color_to_gray_weighted("im1.png", "output_grayscale_image.bmp")



