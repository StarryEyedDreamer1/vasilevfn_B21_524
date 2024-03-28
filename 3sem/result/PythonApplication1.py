import numpy as np
from PIL import Image

image = Image.open('photo_2024-03-21_21-20-49.png')
image_array = np.array(image)

def apply_median_filter(image_array, kernel):
    filtered_image_array = np.zeros_like(image_array)
    for i in range(1, image_array.shape[0] - 1):
        for j in range(1, image_array.shape[1] - 1):
            neighbors = [image_array[i-1, j-1], image_array[i-1, j], image_array[i-1, j+1],
                         image_array[i, j-1], image_array[i, j], image_array[i, j+1],
                         image_array[i+1, j-1], image_array[i+1, j], image_array[i+1, j+1]]
            neighbors.sort()
            filtered_image_array[i, j] = neighbors[4]
    return filtered_image_array

hill_kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
filtered_image_array_hill = apply_median_filter(image_array, hill_kernel)

valley_kernel = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]])
filtered_image_array_valley = apply_median_filter(image_array, valley_kernel)

filtered_image_hill = Image.fromarray(filtered_image_array_hill)
filtered_image_valley = Image.fromarray(filtered_image_array_valley)
filtered_image_hill.show()
filtered_image_valley.show()

filtered_image_hill.save('filtered_image_hill.png')
filtered_image_valley.save('filtered_image_valley.png')