
# Import libraries
import torch
import torch.nn as nn
from torchvision import transforms

# Define the network
class Generator(nn.Module):
    """Generator network."""

    def __init__(self, img_shape=(64, 64), num_channels=1, conv_features=8, num_layers=3, latent_features=128):
        super().__init__()

        # Set the parameters
        self.img_shape = img_shape
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.num_features = conv_features
        self.latent_features = latent_features

        # Calculate variables
        pre_latent_shape = (img_shape[0] // 2**num_layers, img_shape[1] // 2**num_layers)
        pre_latent_features = conv_features * 2**num_layers
        pre_latent_features_total = pre_latent_features * pre_latent_shape[0] * pre_latent_shape[1]

        # Input block
        self.input = nn.Sequential(
            nn.Conv2d(num_channels, conv_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_features),
            nn.ReLU(),
        )

        # Encoder block
        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            num_in = conv_features * 2**i
            num_out = conv_features * 2**(i+1)
            self.encoder.append(nn.Sequential(
                nn.Conv2d(num_in, num_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(num_out),
                nn.ReLU(),
                nn.Conv2d(num_out, num_out, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_out),
                nn.ReLU(),
            ))
        self.encoder.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(pre_latent_features_total, latent_features),
            nn.ReLU(),
            nn.BatchNorm1d(latent_features),
        ))

        # Decoder block
        self.decoder = nn.ModuleList()
        self.decoder.append(nn.Sequential(
            nn.Linear(latent_features, pre_latent_features_total),
            nn.ReLU(),
            nn.BatchNorm1d(pre_latent_features_total),
            nn.Unflatten(-1, (pre_latent_features, *pre_latent_shape)),
        ))
        for i in range(num_layers):
            num_in = conv_features * 2**(num_layers-i)
            num_out = conv_features * 2**(num_layers-i-1)
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(num_in, num_out, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(num_out),
                nn.ReLU(),
                nn.Conv2d(num_out, num_out, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_out),
                nn.ReLU(),
            ))

        # Output block
        self.output = nn.Sequential(
            nn.Conv2d(conv_features, num_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def numel(self):
        """Number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, kT=None):
        """Forward pass."""

        # Input
        x = self.input(x)

        # Encoder
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)

        # Thermal kick
        if kT is not None:
            x = x + kT * torch.randn_like(x)

        # Decoder
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)

        # Output
        x = self.output(x)

        return x
    
    def dream(self, x, kT=None, sigma=0.1, num_steps=100):
        """Dream."""
        
        # Loop over the number of steps
        for i in range(num_steps):

            # Diffuse
            x = x + sigma * torch.randn_like(x)
            x = self.forward(x, kT=kT)

        # Return the result
        return x


    

if __name__ == '__main__':
    # Create a generator
    generator = Generator()

    # Create a random input
    x = torch.randn(1, 1, 64, 64)

    # Generate an image
    out = generator(x)

    # Dream a new image
    out = generator.dream(x)

    # Done!
    print("Done!")

