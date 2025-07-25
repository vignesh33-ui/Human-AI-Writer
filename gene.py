import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

image_size = 28
latent_dim = 100
num_classes = 26
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Generator (must match above)
# ----------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, image_size * image_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.model(gen_input)
        return out.view(out.size(0), 1, image_size, image_size)

# ----------------------------
# Load model
# ----------------------------
generator = Generator().to(device)
generator.load_state_dict(torch.load("cgan_generator.pth", map_location=device))
generator.eval()

# ----------------------------
# Generate word
# ----------------------------
# Define the mapping for letters
class_to_label = {chr(ord('A') + i): i for i in range(26)}
label_to_class = {i: chr(ord('A') + i) for i in range(26)}

def generate_word(word):
    generator.load_state_dict(torch.load('cgan_generator.pth', map_location=device))
    generator.eval()

    images = []
    for char in word.upper():
        if char not in class_to_label:
            print(f"Character {char} not in dataset.")
            continue

        label = torch.tensor([class_to_label[char]], device=device)
        z = torch.randn(1, latent_dim, device=device)
        with torch.no_grad():
            generated_img = generator(z, label).detach().cpu()

        # Ensure 3D: [1, 28, 28]
        if generated_img.ndim == 4:
            generated_img = generated_img.squeeze(0)
        elif generated_img.ndim == 2:
            generated_img = generated_img.unsqueeze(0)

        images.append(generated_img)

    if not images:
        print("No valid characters to generate.")
        return

    combined = torch.cat(images, dim=2)  # concat across width
    img = transforms.ToPILImage()(combined.squeeze(0))
    img.save("generated_word.png")
    img.show()



# Prompt user
user_input = input("Enter a word: ")
generate_word(user_input)
