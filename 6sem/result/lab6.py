import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageOps

TEXT_TO_RENDER = "ԳԵՂԱՐԱՆԻՔ ԴԻԼՈՒՆԸ ԻՄ ԿԱՅՔՈՒՄ ՍԱՍՏԱԾԵՐԸ"
BACKGROUND = 255
FONT_FILE = "Unicode.ttf"
FONT_SIZE = 52
THRESHOLD_VALUE = 75

def load_font(font_path, size):
    return ImageFont.truetype(font_path, size)

def create_output_directories():
    os.makedirs("output/profile_data", exist_ok=True)

def render_text_image(text, font, space_between_chars=5):
    text_width = sum(font.getbbox(char)[2] - font.getbbox(char)[0] for char in text) + space_between_chars * (len(text) - 1)
    max_height = max(font.getbbox(char)[3] for char in text)
    
    text_img = Image.new("L", (text_width, max_height), color="white")
    drawer = ImageDraw.Draw(text_img)
    
    current_x = 0
    for character in text:
        bounding_box = font.getbbox(character)
        char_width = bounding_box[2] - bounding_box[0]
        char_height = bounding_box[3]
        drawer.text((current_x, max_height - char_height), character, "black", font=font)
        current_x += char_width + space_between_chars

    return np.array(text_img)

def binarize_image(image_array, threshold=THRESHOLD_VALUE):
    binary_image = np.zeros(image_array.shape, dtype=np.uint8)
    binary_image[image_array > threshold] = BACKGROUND
    return binary_image

def save_profiles(image_array):
    os.makedirs("output/profile_data", exist_ok=True)
    binary_image = np.where(image_array != BACKGROUND, 1, 0)
    
    plt.bar(np.arange(1, binary_image.shape[1] + 1), np.sum(binary_image, axis=0), width=0.9)
    plt.savefig("output/profile_data/horizontal_profile.png")
    plt.clf()
    
    plt.barh(np.arange(1, binary_image.shape[0] + 1), np.sum(binary_image, axis=1), height=0.9)
    plt.savefig("output/profile_data/vertical_profile.png")
    plt.clf()

def identify_letter_boundaries(binary_image):
    profile = np.sum(binary_image == 0, axis=0)
    letter_intervals = []
    inside_letter = False

    for idx, pixel_count in enumerate(profile):
        if pixel_count > 0:
            if not inside_letter:
                inside_letter = True
                start_idx = idx
        else:
            if inside_letter:
                inside_letter = False
                end_idx = idx
                letter_intervals.append((start_idx - 1, end_idx))
    
    if inside_letter:
        letter_intervals.append((start_idx, len(profile)))
    
    return letter_intervals

def outline_letters(image_array, boundaries):
    outlined_image = Image.fromarray(image_array)
    drawer = ImageDraw.Draw(outlined_image)
    
    for start, end in boundaries:
        drawer.rectangle([start, 0, end, image_array.shape[0]], outline="red")
    
    outlined_image.save("output/outlined_text.bmp")

if __name__ == "__main__":
    create_output_directories()
    font = load_font(FONT_FILE, FONT_SIZE)
    rendered_text_image = render_text_image(TEXT_TO_RENDER, font)
    binary_image = binarize_image(rendered_text_image)
    binarized_img = Image.fromarray(binary_image)
    binarized_img.save("output/rendered_text.bmp")
    ImageOps.invert(binarized_img).save("output/inverted_text.bmp")
    save_profiles(binary_image)
    letter_bounds = identify_letter_boundaries(binary_image)
    outline_letters(binary_image, letter_bounds)
