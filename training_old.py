
# Import libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from model import Generator
from torch.utils.data import DataLoader
from data_generator import ChineseCharacters

# Set environment
device = torch.device("cpu")

# Set constants
num_epochs = 10000
num_batches = 10
batch_size = 32
image_size = (64, 64)

# Set up the DataLoader for the dataset
dataset = ChineseCharacters(image_size=image_size)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the generator model and the loss function
generator = Generator()
generator_path = "models/generator_auto1.pth"
try:
    generator.load_state_dict(torch.load(generator_path))
    print("Generator found. Loading weights.")
except:
    print("Generator not found. Training from scratch.")
optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
def loss_fn(output_data, target_data):
    mask = target_data.detach().numpy() < .5
    border = np.zeros(mask.shape, dtype=bool)
    for i in range(border.shape[0]):
        border[i, 0, :, :] = segmentation.find_boundaries(mask[i, 0, :, :], mode='thick')
    loss = (
        F.mse_loss(output_data, target_data)
        + 2 * F.mse_loss(output_data[mask], target_data[mask])
        + 3 * F.mse_loss(output_data[border], target_data[border])
    )
    return loss


# Train the model
for epoch in range(num_epochs):

    # Iterate over the batches
    batch_idx = 0
    for target_data in train_loader:
        batch_idx += 1
        if batch_idx >= num_batches:
            break

        # Set constants
        num_steps = 1 + np.random.poisson(1)
        temperature = .1 + np.random.exponential(.05, size=batch_size).astype(np.float32)

        # Set model to training mode
        generator.train()
        optimizer.zero_grad()

        # Generate corrupted images
        with torch.no_grad():
            input_data = generator.corrupt_image(target_data, num_steps=num_steps, temperature=temperature)
            input_data -= input_data.min()
            if input_data.max() > 0:
                input_data /= input_data.max()

        # Pass the corrupted images through the generator
        output_data = generator.diffusion(input_data, num_steps=num_steps, temperature=temperature)

        # Compute the loss
        loss = loss_fn(output_data, target_data)

        # Backpropagate the loss
        loss.backward()

        # Update the weights
        optimizer.step()

        # Print progress
        if (batch_idx % (num_batches//10) == 0) or (batch_idx < 3):
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")
            
            # Set up the figure
            fig = plt.gcf()
            fig.clf()
            plt.ion()
            plt.show()
            ax = np.empty((1, 3), dtype=object)
            for i in range(ax.shape[0]):
                for j in range(ax.shape[1]):
                    ax[i, j] = fig.add_subplot(ax.shape[0], ax.shape[1], i*ax.shape[1] + j + 1)

            # Set up the colorbar
            divider = make_axes_locatable(ax[0, -1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cmin = np.min([target_data.min().item(), input_data.min().item(), output_data.min().item()])
            cmax = np.max([target_data.max().item(), input_data.max().item(), output_data.max().item()])

            # Plot the images
            im = ax[0, 0].imshow(target_data[0, 0, :, :].detach().numpy(), cmap='gray', vmin=cmin, vmax=cmax)
            ax[0, 0].set_title("Target")
            ax[0, 1].imshow(input_data[0, 0, :, :].detach().numpy(), cmap='gray', vmin=cmin, vmax=cmax)
            ax[0, 1].set_title("Corrupted")
            ax[0, 2].imshow(output_data[0, 0, :, :].detach().numpy(), cmap='gray', vmin=cmin, vmax=cmax)
            ax[0, 2].set_title("Output")

            # Finalize
            fig.suptitle(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{num_batches}, num_steps={num_steps}")
            fig.colorbar(im, ax=ax.ravel().tolist(), cax=cax, orientation='vertical')
            plt.tight_layout()
            plt.pause(.1)
        
    # Save the model
    torch.save(generator.state_dict(), generator_path)

# Dream new images
fig, ax = plt.subplots(1, 1)
plt.ion()
plt.show()
with torch.no_grad():
    x = torch.randn(1, 1, 64, 64)
    ax.imshow(x[0, 0, :, :].detach().numpy(), cmap='gray')
    ax.set_title(f"Initial")
    plt.pause(1)
    for i in range(100):
        x = generator.diffusion(x, num_steps=10, temperature=.1)
        ax.cla()
        ax.imshow(x[0, 0, :, :].detach().numpy(), cmap='gray')
        ax.set_title(f"Step {i}/100")
        plt.pause(.1)

