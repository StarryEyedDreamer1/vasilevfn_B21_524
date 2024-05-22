import csv
import math
import numpy as np
from PIL import Image

# Константы для работы с изображениями и текстом
UNICODE_LETTERS = ["0531", "0532", "0533", "0534", "0535", "0536", "0537", "0538", "0539", "053A", "053B",
                   "053C", "053D", "053E", "053F", "0540", "0541", "0542", "0543", "0544", "0545", "0546",
                   "0547", "0548", "0549", "054A", "054B", "054C", "054D", "054E", "054F", "0550", "0551",
                   "0552", "0553", "0554", "0555", "0556"]
LETTER_SYMBOLS = [chr(int(code, 16)) for code in UNICODE_LETTERS]

BACKGROUND_COLOR = 255
TARGET_TEXT = "ԳԵՂԱՐԱՆԻՔ ԴԻԼՈՒՆԸ ԻՄ ԿԱՅՔՈՒՄ ՍԱՍՏԱԾԵՐԸ".replace(" ", "")

def extract_features(image_array: np.array):
    binary_image = np.where(image_array != BACKGROUND_COLOR, 1, 0)
    total_weight = np.sum(binary_image)

    y_coords, x_coords = np.indices(binary_image.shape)
    y_center = np.sum(y_coords * binary_image) / total_weight
    x_center = np.sum(x_coords * binary_image) / total_weight
    
    inertia_x = np.sum((y_coords - y_center) ** 2 * binary_image) / total_weight
    inertia_y = np.sum((x_coords - x_center) ** 2 * binary_image) / total_weight

    return total_weight, x_center, y_center, inertia_x, inertia_y

def find_letter_segments(image_array):
    vertical_profile = np.sum(image_array == 0, axis=0)
    segments = []
    is_letter = False

    for index, value in enumerate(vertical_profile):
        if value > 0:
            if not is_letter:
                is_letter = True
                start_index = index
        else:
            if is_letter:
                is_letter = False
                end_index = index
                segments.append((start_index - 1, end_index))

    if is_letter:
        segments.append((start_index, len(vertical_profile)))

    return segments

def load_alphabet_data() -> dict:
    def parse_str_to_tuple(string):
        return tuple(map(float, string.strip('()').split(',')))

    alphabet_data = {}
    with open('data.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            weight = int(row['weight'])
            center_of_mass = parse_str_to_tuple(row['center_of_mass'])
            inertia = parse_str_to_tuple(row['inertia'])
            alphabet_data[LETTER_SYMBOLS[index]] = (weight, *center_of_mass, *inertia)
    return alphabet_data

def compute_similarity(alphabet_data: dict, letter_features):
    def euclidean_distance(feature1, feature2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(feature1, feature2)))

    distances = {letter: euclidean_distance(letter_features, features) for letter, features in alphabet_data.items()}
    max_distance = max(distances.values())
    
    similarities = [(letter, round(1 - distance / max_distance, 2)) for letter, distance in distances.items()]
    return sorted(similarities, key=lambda x: x[1])

def recognize_text(image_array: np.array, letter_segments):
    alphabet_data = load_alphabet_data()
    recognized_letters = []

    for start, end in letter_segments:
        letter_image = image_array[:, start:end]
        letter_features = extract_features(letter_image)
        hypotheses = compute_similarity(alphabet_data, letter_features)
        best_match = hypotheses[-1][0]
        recognized_letters.append(best_match)

    return "".join(recognized_letters)

def save_results(recognized_text: str):
    max_length = max(len(TARGET_TEXT), len(recognized_text))
    original_padded = TARGET_TEXT.ljust(max_length)
    recognized_padded = recognized_text.ljust(max_length)

    with open("output/results.txt", 'w', encoding='utf-8') as file:
        correct_count = 0
        comparison_lines = ["Expected | Detected | Match"]
        for i in range(max_length):
            match = original_padded[i] == recognized_padded[i]
            comparison_lines.append(f"{original_padded[i]} | {recognized_padded[i]} | {match}")
            correct_count += int(match)
        
        accuracy = math.ceil(correct_count / max_length * 100)
        file.write(f"Original:   {original_padded}\n")
        file.write(f"Recognized: {recognized_padded}\n")
        file.write(f"Accuracy:   {accuracy}%\n\n")
        file.write("\n".join(comparison_lines))

if __name__ == "__main__":
    image = np.array(Image.open('original_phrase.bmp').convert('L'))
    segments = find_letter_segments(image)
    recognized_text = recognize_text(image, segments)
    save_results(recognized_text)

