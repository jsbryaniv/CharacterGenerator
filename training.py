
# Import libraries
import os
import time
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur
from model import Generator
from torch.utils.data import DataLoader
from data_generator import ChineseCharacters

# Set environment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

# Set constants
image_size = (64, 64)
lr = 0.001
batch_size = 32
num_batches = 100
num_epochs = 1000
num_steps = 5  # Number of times to add noise to the input image

# Set variables
# noise_levels = [2**(i+2-num_steps) for i in range(num_steps)]
noise_levels = np.linspace(.1, 2, num_steps)

# Image viewer
def view(image, id=0):
    plt.imshow(image.detach().cpu().numpy()[id, 0, :, :], cmap='gray')
    plt.pause(.1)
    plt.show()
    return

# Results viewer
def view_results(inputs, outputs):
    
    # Set up figure
    fig = plt.gcf()
    plt.ion()
    plt.clf()
    plt.show()
    num_images = len(inputs)
    ax = np.empty((2, num_images), dtype=object)
    for i in range(num_images):
        ax[0, i] = fig.add_subplot(2, num_images, i+1)
        ax[1, i] = fig.add_subplot(2, num_images, num_images+i+1)
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
    ax[0, 0].set_ylabel("Input")
    ax[1, 0].set_ylabel("Output")
    
    # Loop over images
    for i in range(num_images):
        ax[0, i].set_title(f"Step {i+1}")
        ax[0, i].imshow(inputs[i].detach().cpu().numpy()[0, 0, :, :], cmap='gray')
        ax[1, i].imshow(outputs[i].detach().cpu().numpy()[0, 0, :, :], cmap='gray')

    # Finalize figure
    plt.tight_layout()
    plt.pause(.1)
    
    # Return
    return fig, ax



### Set up Data ###

# Set up the DataLoader for the dataset
dataset = ChineseCharacters(image_size=image_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


### Set up Model ###

# Define the generator model and the loss function
generator = Generator()
generator_path = "models/generator_auto1.pth"
try:
    # Try to load model weights
    with open(generator_path, 'rb') as f:
        generator.load_state_dict(torch.load(f))
    print(f"Model found. Loading weights.")
except:
    # Initialize parameters to small random values
    for param in generator.parameters():
        param.data = torch.randn(param.shape) * .01
    print(f"Model not found. Training from scratch.")
generator.to(device)

# Set up the optimizer
optimizer = torch.optim.Adam(generator.parameters(), lr=lr)


### Train Model ###

# Loop over epochs
print("Training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch} of {num_epochs}")

    # Loop over batches
    batch_idx = 0
    while batch_idx < num_batches:
        batch_idx += 1
        t = time.time()

        # Get images
        target_image = next(iter(dataloader)).to(device)
        image = target_image.clone()
        input_images = []
        output_images = []

        # Set up model
        generator.train()
        generator.zero_grad()

        # Initialize loss with regulator
        loss = (
            sum((p-.01).pow(2.0).sum() for p in generator.parameters())
            / sum(p.numel() for p in generator.parameters())
        )

        # Train pure autoencoder
        for i, sigma in enumerate(noise_levels):
            
            # Add noise
            image = target_image.clone() + torch.randn_like(image, device=device) * sigma

            # Forward pass
            image = generator(image)

            # Calculate loss
            loss += F.mse_loss(image, target_image)

        # Train pyramid model
        image = target_image.clone()
        for i, sigma in enumerate(noise_levels):
            last_loop = i == len(noise_levels) - 1
            
            # Add noise
            image = image + torch.randn_like(image, device=device) * sigma
            input_images.append(image.clone())

            # Forward pass
            image = generator(image)
            output_images.append(image.clone())

            # Calculate loss
            loss += F.mse_loss(image, target_image)

            # Loop down the pyramid
            image2 = image.clone()
            for j, sigma2 in enumerate(noise_levels[:i][::-1]):

                # Add noise
                image2 = image + torch.randn_like(image2, device=device) * sigma2
                if last_loop:
                    input_images.append(image2.clone())

                # Forward pass
                image2 = generator(image2)
                if last_loop:
                    output_images.append(image2.clone())

                # Calculate loss
                loss += F.mse_loss(image2, target_image)

        # Backpropagate
        loss.backward()
        optimizer.step()

        # Print update
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} of {num_batches}: loss = {loss.item():.4f} ({(time.time()-t):.2f} s)")
            view_results(input_images, output_images)

    # Dream
    generator.eval()
    image = torch.zeros((1, 1, *image_size), device=device)
    image = generator.dream(image, kT=0, sigma=1.0, num_steps=100)
    plt.clf()
    view(image)
    
    # Save model
    with open(generator_path, 'wb') as f:
        torch.save(generator.state_dict(), f)
        
# Done
print("Finished training!")

