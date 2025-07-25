import os
import xml.etree.ElementTree as ET
from PIL import Image
import string

# --- Configuration ---
ANNOTATIONS_DIR = "anno"     # Where XML files are
IMAGES_DIR = "images"               # Full input images (PNG/JPG)
OUTPUT_DIR = "dataset"              # Where cropped letters will be saved

# --- Create folders A-Z ---
for letter in string.ascii_uppercase:
    os.makedirs(os.path.join(OUTPUT_DIR, letter), exist_ok=True)

# --- Count existing files to auto-increment ---
letter_counts = {letter: len(os.listdir(os.path.join(OUTPUT_DIR, letter))) for letter in string.ascii_uppercase}

# --- Process all XML files ---
for xml_file in os.listdir(ANNOTATIONS_DIR):
    if not xml_file.endswith('.xml'):
        continue

    xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_file = root.find('filename').text
    image_path = os.path.join(IMAGES_DIR, image_file)

    if not os.path.exists(image_path):
        print(f"Image not found for: {image_file}")
        continue

    # Open the image
    image = Image.open(image_path).convert('L')  # grayscale

    # Loop through each labeled object
    for obj in root.findall('object'):
        label = obj.find('name').text.strip().upper()

        if label not in string.ascii_uppercase:
            print(f"Skipping non-capital label: {label}")
            continue

        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        # Crop the letter
        cropped_letter = image.crop((xmin, ymin, xmax, ymax))
        resized_letter = cropped_letter.resize((28, 28))

        # Save as dataset/A/A_1.png, etc.
        letter_counts[label] += 1
        filename = f"{label}_{letter_counts[label]}.png"
        output_path = os.path.join(OUTPUT_DIR, label, filename)

        resized_letter.save(output_path)
        print(f"Saved: {output_path}")
