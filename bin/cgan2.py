import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import string
import random

# Load your trained CGAN generator
from model import Generator  # make sure this is your generator class
generator = Generator()
generator.load_state_dict(torch.load("generator.pth"))  # path to saved generator
generator.eval()

# Settings
img_size = 64  # size of each letter (must match training)
latent_dim = 100
word = "HELLO"  # the word you want to generate
canvas_width = len(word) * (img_size + 5)
canvas_height = img_size
letter_spacing = 5  # space between letters

# Create blank canvas
canvas = Image.new("L", (canvas_width, canvas_height), 255)

# Function to generate letter
def generate_letter(letter):
    letter_idx = string.ascii_uppercase.index(letter.upper())
    z = torch.randn(1, latent_dim)
    label = torch.tensor([letter_idx])
    gen_image = generator(z, label)
    image_np = gen_image.detach().cpu().numpy().squeeze()
    image_np = (image_np * 127.5 + 127.5).astype(np.uint8)  # Rescale from [-1, 1] to [0, 255]
    return Image.fromarray(image_np)

# Stitch letters together
x_offset = 0
for char in word:
    letter_img = generate_letter(char)
    
    # Optional: Add some slight random offsets to simulate imperfection
    jitter_x = random.randint(-2, 2)
    jitter_y = random.randint(-2, 2)

    canvas.paste(letter_img, (x_offset + jitter_x, jitter_y))
    x_offset += img_size + letter_spacing

# Save or show
canvas.save("generated_word.png")
canvas.show()
