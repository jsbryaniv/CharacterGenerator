
# Import libraries
import os
import time
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
num_epochs = 20



### Set up Data ###

# Set up the DataLoader for the dataset
dataset = ChineseCharacters(image_size=image_size, transform=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


### Set up Model ###

# Define the generator model and the loss function
generator = Generator()
generator_path = "models/generator.pth"
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



### Plotting Functions ###

# Image viewer
def view(image, id=0):
    plt.imshow(image.detach().cpu().numpy()[id, 0, :, :], cmap='gray')
    plt.pause(.1)
    plt.show()
    return

# Results viewer
def view_results(inputs, outputs, id=0):
    
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
        ax[0, i].imshow(inputs[i].detach().cpu().numpy()[id, 0, :, :], cmap='gray')
        ax[1, i].imshow(outputs[i].detach().cpu().numpy()[id, 0, :, :], cmap='gray')
        ax[1, i].set_xlabel(f"Step {i+1}", rotation=45)

    # Finalize figure
    plt.tight_layout()
    plt.pause(.1)
    
    # Return
    return fig, ax


### Train Model ###

# Loop over epochs
print(f"Training generator with {generator.numel()} parameters on {device}")
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
            sum(p.pow(2.0).sum() for p in generator.parameters())
            / sum(p.numel() for p in generator.parameters())
        )

        # Increase temperature
        image = target_image.clone()
        for i, kT in enumerate(generator.kT_schedule):

            # Check if end of schedule
            last_step = i == generator.num_steps - 1
            
            # Add noise
            image = image + torch.randn_like(image, device=device) * kT
            input_images.append(image.clone())

            # Forward pass
            image = generator(image)
            output_images.append(image.clone())
            
            # Loop down temperature schedule from i to 0
            for j in range(i-1, -1, -1):
                kT2 = generator.kT_schedule[j]
                
                # Add noise
                image = image + torch.randn_like(image, device=device) * kT2
                if last_step:
                    input_images.append(image.clone())
                
                # Forward pass
                image = generator(image)
                if last_step:
                    output_images.append(image.clone())

            # Add loss
            loss += F.mse_loss(image, target_image)

        # Backpropagate
        loss.backward()
        optimizer.step()

        # Print update
        if batch_idx % 10 == 0:
            status = (
                f"Batch {batch_idx} of {num_batches}:"
                + f" loss = {loss.item():.4f} ({(time.time()-t):.2f} s)"
            )
            print(status)
            view_results(input_images, output_images)

    # Dream
    generator.eval()
    image = generator.max_kT * torch.randn((1, 1, *image_size), device=device)
    image = next(iter(dataloader)).to(device)[[0]]
    for _ in range(10):
        image = generator.dream(image)
    plt.clf()
    view(image)
    
    # Save model
    with open(generator_path, 'wb') as f:
        torch.save(generator.state_dict(), f)
        
# Done
print("Finished training!")

