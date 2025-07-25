import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# ----------------------------
# CONFIG
# ----------------------------
image_size = 28
batch_size = 28
latent_dim = 100
epochs = 5000
lr = 0.002
num_classes = 26  # A-Z
sample_dir = "samples"
os.makedirs(sample_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Dataset
# ----------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = ImageFolder(root="dataset", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ----------------------------
# Generator
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
# Discriminator
# ----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(num_classes + image_size * image_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        d_input = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_input)
        return validity

# ----------------------------
# Initialize & Optimizers
# ----------------------------
generator = Generator().to(device)
discriminator = Discriminator().to(device)
loss_fn = nn.BCELoss()
opt_G = torch.optim.Adam(generator.parameters(), lr=lr)
opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.size(0)
        real = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        imgs, labels = imgs.to(device), labels.to(device)

        # -----------------
        # Train Generator
        # -----------------
        noise = torch.randn(batch_size, latent_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
        gen_imgs = generator(noise, gen_labels)

        g_loss = loss_fn(discriminator(gen_imgs, gen_labels), real)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        # ---------------------
        # Train Discriminator
        # ---------------------
        real_loss = loss_fn(discriminator(imgs, labels), real)
        fake_loss = loss_fn(discriminator(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        if i % 100 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    # Save generated images each epoch
    save_image(gen_imgs.data[:25], f"{sample_dir}/{epoch}.png", nrow=5, normalize=True)

# Save generator model
torch.save(generator.state_dict(), "cgan_generator.pth")
print("✅ Model saved as cgan_generator.pth")
