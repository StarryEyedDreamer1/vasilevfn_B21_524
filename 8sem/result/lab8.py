import os
import typing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

# Константы
WHITE_PIXEL_VALUE = 255
DISTANCE = 1
ANGLES_RAD = [np.deg2rad(angle) for angle in [0, 90, 180, 270]]

# Тип данных для координат точки
Coordinate = typing.Tuple[int, int]

def analyze_neighbors(image: Image, pixel_data, position: Coordinate):
    neighbor_counts = np.zeros(WHITE_PIXEL_VALUE + 1)
    x_base, y_base = position
    for angle in ANGLES_RAD:
        x_neighbor = int(x_base + np.round(np.cos(angle)) * DISTANCE)
        y_neighbor = int(y_base + np.round(np.sin(angle)) * DISTANCE)
        if 0 <= x_neighbor < image.size[0] and 0 <= y_neighbor < image.size[1]:
            neighbor_value = pixel_data[x_neighbor, y_neighbor]
            neighbor_counts[neighbor_value] += 1
    return pixel_data[x_base, y_base], neighbor_counts

def generate_pixels(image: Image, function=lambda img, pix, pos: pix[pos]):
    pixels = image.load()
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            position = (x, y)
            yield position, function(image, pixels, position)

def create_haralick_matrix(image_name: str, grayscale_image: Image):
    matrix = np.zeros((WHITE_PIXEL_VALUE + 1, WHITE_PIXEL_VALUE + 1))
    histogram = np.zeros(WHITE_PIXEL_VALUE + 1)
    max_value = 0
    total_pixels = grayscale_image.size[0] * grayscale_image.size[1]
    
    for pos, (value, counts) in tqdm(generate_pixels(grayscale_image, function=analyze_neighbors), total=total_pixels):
        matrix[value] += counts
        max_value = max(max_value, counts.max())
        histogram[value] += 1
    
    scaled_matrix = np.uint8(matrix * WHITE_PIXEL_VALUE / max_value)
    Image.fromarray(scaled_matrix).save(f"{image_name}_matrix.jpg", "JPEG")

    plot_histogram(histogram, image_name)
    calculate_statistics(matrix, image_name)

def calculate_statistics(haralick_matrix: np.array, output_name: str):
    total_sum = haralick_matrix.sum()
    probability_matrix = haralick_matrix / total_sum
    
    asm = np.sum(probability_matrix ** 2)
    mean_intensity = np.sum(probability_matrix * np.arange(WHITE_PIXEL_VALUE + 1))
    entropy = -np.sum(probability_matrix * np.log2(probability_matrix + np.finfo(float).eps))
    contrast = np.sum((np.arange(WHITE_PIXEL_VALUE + 1)[:, None] - np.arange(WHITE_PIXEL_VALUE)) ** 2 * probability_matrix)
    
    stats = pd.Series({"ASM": asm, "Mean Intensity": mean_intensity, "Entropy": entropy, "Contrast": contrast})
    stats.to_csv(f"{output_name}.csv")

def apply_linear_transformation(image: Image, c=1, b=0):
    transformed_image = image.copy()
    drawer = ImageDraw.Draw(transformed_image)
    for position, pixel_value in generate_pixels(image):
        new_value = min(max(int(c * pixel_value + b), 0), WHITE_PIXEL_VALUE)
        drawer.point(position, new_value)
    return transformed_image

def plot_histogram(histogram, image_name):
    plt.figure()
    plt.bar(np.arange(histogram.size), histogram)
    plt.savefig(f"{image_name}_histogram.png")
    plt.close()

def process_images():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_directory, 'input')
    
    for image_file in os.scandir(input_dir):
        print(f"Processing {image_file.name}...")
        output_dir = os.path.join(current_directory, 'output', image_file.name.split('.')[0])
        os.makedirs(output_dir, exist_ok=True)
        
        original_image_path = os.path.join(output_dir, "original")
        transformed_image_path = os.path.join(output_dir, "transformed")
        
        grayscale_image = Image.open(image_file.path).convert('L')
        grayscale_image.save(f"{original_image_path}.jpg", "JPEG")
        
        create_haralick_matrix(original_image_path, grayscale_image)
        
        transformed_image = apply_linear_transformation(grayscale_image, c=1.5, b=-30)
        transformed_image.save(f"{transformed_image_path}.jpg", "JPEG")
        
        create_haralick_matrix(transformed_image_path, transformed_image)

if __name__ == "__main__":
    process_images()
