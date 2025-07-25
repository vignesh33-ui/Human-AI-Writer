import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import string
import random

# CONFIG
IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 5000
LATENT_DIM = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

letters = string.ascii_uppercase
label_to_idx = {l: i for i, l in enumerate(letters)}

# -------- Dataset --------
class LetterDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label in letters:
            folder = os.path.join(root_dir, label)
            for file in os.listdir(folder):
                if file.endswith(".png"):
                    self.samples.append((os.path.join(folder, file), label_to_idx[label]))
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        return self.transform(img), label

# -------- Generator --------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(len(letters), len(letters))
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + len(letters), 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, IMG_SIZE * IMG_SIZE),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], 1)
        img = self.model(x)
        return img.view(img.size(0), 1, IMG_SIZE, IMG_SIZE)

# -------- Discriminator --------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(len(letters), len(letters))
        self.model = nn.Sequential(
            nn.Linear(IMG_SIZE * IMG_SIZE + len(letters), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        c = self.label_emb(labels)
        x = torch.cat([img_flat, c], 1)
        return self.model(x)

# -------- Train --------
def train():
    dataset = LetterDataset("dataset")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    loss_fn = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(EPOCHS):
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            valid = torch.ones(imgs.size(0), 1, device=DEVICE)
            fake = torch.zeros(imgs.size(0), 1, device=DEVICE)

            # --- Train Generator ---
            z = torch.randn(imgs.size(0), LATENT_DIM, device=DEVICE)
            gen_imgs = G(z, labels)
            g_loss = loss_fn(D(gen_imgs, labels), valid)

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            # --- Train Discriminator ---
            real_loss = loss_fn(D(imgs, labels), valid)
            fake_loss = loss_fn(D(gen_imgs.detach(), labels), fake)
            d_loss = (real_loss + fake_loss) / 2

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        # Sample generation for 'A'
        sample_text(G, "A", epoch + 1)

# -------- Sample Generation --------
def sample_text(generator, letter, epoch):
    generator.eval()
    with torch.no_grad():
        label = torch.tensor([label_to_idx[letter]]).to(DEVICE)
        noise = torch.randn(1, LATENT_DIM).to(DEVICE)
        img = generator(noise, label).cpu().squeeze().numpy()
        img = ((img + 1) * 127.5).astype(np.uint8)
        Image.fromarray(img).save(f"samples/{letter}_{epoch}.png")
    generator.train()

# -------- Run --------
if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True)
    train()
