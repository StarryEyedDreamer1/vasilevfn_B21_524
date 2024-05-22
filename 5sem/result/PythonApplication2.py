import csv
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps
from matplotlib import pyplot as plt

UNICODE_PERSIAN_LETTERS = [
    "0531", "0532", "0533", "0534", "0535", "0536", "0537", "0538", "0539", "053A",
    "053B", "053C", "053D", "053E", "053F", "0540", "0541", "0542", "0543", "0544",
    "0545", "0546", "0547", "0548", "0549", "054A", "054B", "054C", "054D", "054E",
    "054F", "0550", "0551", "0552", "0553", "0554", "0555", "0556"
]

# Преобразование Unicode в символы
LETTERS_PERSIAN = [chr(int(code, 16)) for code in UNICODE_PERSIAN_LETTERS]

# Параметры шрифта и пороговое значение
FONT_SIZE = 52
BINARIZATION_THRESHOLD = 75
FONT_FILE_PATH = "Unicode.ttf"
PIXEL_WHITE = 255

def binarize_image(img, threshold=BINARIZATION_THRESHOLD):
    grayscale_image = (0.3 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 2]).astype(np.uint8)
    binary_image = np.zeros_like(grayscale_image)
    binary_image[grayscale_image > threshold] = PIXEL_WHITE
    return binary_image.astype(np.uint8)

def create_letter_images(letters):
    font = ImageFont.truetype(FONT_FILE_PATH, FONT_SIZE)
    os.makedirs("output/letters", exist_ok=True)
    os.makedirs("output/inverted_letters", exist_ok=True)

    for idx, letter in enumerate(letters):
        _, _, _, bottom = font.getbbox(letter)
        img_height = bottom
        letter_img = Image.new("RGB", (FONT_SIZE, img_height), "white")
        draw = ImageDraw.Draw(letter_img)
        draw.text((0, 0), letter, "black", font=font)
        binary_img = Image.fromarray(binarize_image(np.array(letter_img)), 'L')
        binary_img.save(f"output/letters/{idx + 1}.png")
        ImageOps.invert(binary_img).save(f"output/inverted_letters/{idx + 1}.png")

def extract_image_features(image):
    binary = np.where(image != PIXEL_WHITE, 1, 0)
    height, width = binary.shape
    half_height, half_width = height // 2, width // 2

    regions = {
        'upper_left': binary[:half_height, :half_width],
        'upper_right': binary[:half_height, half_width:],
        'lower_left': binary[half_height:, :half_width],
        'lower_right': binary[half_height:, half_width:]
    }

    weights = {k: np.sum(v) for k, v in regions.items()}
    normalized_weights = {k: v / (half_height * half_width) for k, v in weights.items()}
    total_weight = np.sum(binary)
    
    y_indices, x_indices = np.indices(binary.shape)
    center_y = np.sum(y_indices * binary) / total_weight
    center_x = np.sum(x_indices * binary) / total_weight
    center_of_mass = (center_x, center_y)
    norm_center_of_mass = (center_x / (width - 1), center_y / (height - 1))
    
    inertia_y = np.sum((x_indices - center_x) ** 2 * binary) / total_weight
    inertia_x = np.sum((y_indices - center_y) ** 2 * binary) / total_weight
    norm_inertia_y = inertia_y / height ** 2
    norm_inertia_x = inertia_x / width ** 2

    return {
        'total_weight': total_weight,
        'weights': weights,
        'normalized_weights': normalized_weights,
        'center_of_mass': center_of_mass,
        'norm_center_of_mass': norm_center_of_mass,
        'inertia': (inertia_x, inertia_y),
        'norm_inertia': (norm_inertia_x, norm_inertia_y)
    }

def save_features_to_csv(letters):
    os.makedirs('output', exist_ok=True)
    with open('output/features.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['total_weight', 'weights', 'normalized_weights', 'center_of_mass', 'norm_center_of_mass', 'inertia', 'norm_inertia']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx in range(len(letters)):
            img = np.array(Image.open(f'output/letters/{idx + 1}.png').convert('L'))
            features = extract_image_features(img)
            writer.writerow(features)

def create_image_profiles(letters):
    os.makedirs("output/profiles/horizontal", exist_ok=True)
    os.makedirs("output/profiles/vertical", exist_ok=True)

    for idx in range(len(letters)):
        img = np.array(Image.open(f'output/letters/{idx + 1}.png').convert('L'))
        binary = np.where(img != PIXEL_WHITE, 1, 0)

        plt.bar(np.arange(1, binary.shape[1] + 1), np.sum(binary, axis=0), width=0.9)
        plt.ylim(0, FONT_SIZE)
        plt.xlim(0, 55)
        plt.savefig(f'output/profiles/horizontal/{idx + 1}.png')
        plt.clf()

        plt.barh(np.arange(1, binary.shape[0] + 1), np.sum(binary, axis=1), height=0.9)
        plt.ylim(FONT_SIZE, 0)
        plt.xlim(0, 55)
        plt.savefig(f'output/profiles/vertical/{idx + 1}.png')
        plt.clf()

if __name__ == "__main__":
    create_letter_images(LETTERS_PERSIAN)
    save_features_to_csv(LETTERS_PERSIAN)
    create_image_profiles(LETTERS_PERSIAN)
