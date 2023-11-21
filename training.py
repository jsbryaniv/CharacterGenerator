
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
blur_width = .02*max(image_size)  # Standard deviation of the Gaussian blur kernel
blur_levels = [blur_width * 2**i for i in range(3)]
noise_levels = [2**(i+1-num_steps) for i in range(num_steps)]
noise_levels = noise_levels + [2] + noise_levels[::-1]

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
        param.data = torch.randn(param.shape) * .1
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

        # Get batch
        target_image = next(iter(dataloader)).to(device)

        # Set up model
        generator.train()
        generator.zero_grad()

        # Initialize loss
        loss = 0

        # Train single image distortion
        for sigma in noise_levels:
            for blur_scale in blur_levels:
                image = target_image
                image = GaussianBlur(kernel_size=9, sigma=blur_scale)(image)
                image = image + torch.randn_like(image, device=device) * sigma
                image = generator(image)
                loss += F.mse_loss(image, target_image)

        # Train running image distortion
        image = target_image.clone()
        input_images = []
        output_images = []
        for sigma in noise_levels:
            
            # Add noise
            image = GaussianBlur(kernel_size=9, sigma=blur_width)(image)
            image = image + torch.randn_like(image, device=device) * sigma
            input_images.append(image.clone())

            # Forward pass
            image = generator(image)
            output_images.append(image.clone())

            # Calculate loss
            loss += F.mse_loss(image, target_image)

        # Backpropagate
        loss.backward()
        optimizer.step()

        # Print update
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} of {num_batches}: loss = {loss.item():.4f} ({(time.time()-t):.2f} s)")
            view_results(input_images, output_images)

    # Dream
    generator.eval()
    image = torch.randn((1, 1, *image_size), device=device)
    for i in range(100):
        image = generator.dream(image, kT=1, sigma=0.1, num_steps=10)
        image = generator.dream(image, kT=2, sigma=0.0, num_steps=10)
        plt.clf()
        view(image)
    
    # Save model
    with open(generator_path, 'wb') as f:
        torch.save(generator.state_dict(), f)
        
# Done
print("Finished training!")

