
# Import libraries
import torch
import torch.nn as nn
from torchvision import transforms


# Define the network
class Generator(nn.Module):
    """Generator network."""

    def __init__(self, img_shape=(64, 64), num_channels=1, num_features=8, num_layers=4, num_steps=8, max_kT=1.5):
        super().__init__()

        # Set the parameters
        self.img_shape = img_shape
        self.num_channels = num_channels
        self.num_features = num_features
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.max_kT = max_kT
        
        # Set temperature schedule
        self.kT_schedule = list(torch.linspace(0, max_kT, num_steps))


        # Input block
        self.input = nn.Sequential(
            nn.Conv2d(num_channels, num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
        )

        # Encoder block
        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            num_in = num_features * 2**i
            num_out = num_features * 2**(i+1)
            self.encoder.append(nn.Sequential(
                nn.Conv2d(num_in, num_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(num_out),
                nn.ReLU(),
                nn.Conv2d(num_out, num_out, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_out),
                nn.ReLU(),
            ))

        # Decoder block
        self.decoder = nn.ModuleList()
        for i in range(num_layers):
            num_in = num_features * 2**(num_layers-i)
            num_out = num_features * 2**(num_layers-i-1)
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
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            nn.Conv2d(num_features, num_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def numel(self):
        """Number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """Forward pass."""

        # Initialze skips
        skips = []

        # Normalize the input
        x = x - x.mean(dim=(2, 3), keepdim=True)
        x = x / x.std(dim=(2, 3), keepdim=True)

        # Input
        x = self.input(x)

        # Encoder
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            skips.append(x)
            
        # Decoder
        for i in range(len(self.decoder)):
            x = x + skips[-i-1]
            x = self.decoder[i](x)

        # Output
        x = self.output(x)

        return x
    
    @torch.no_grad()
    def dream(self, num_cycles=10, kT_schedule=None):
        """Dream."""
        
        # Set the temperature schedule
        if kT_schedule is None:
            kT_schedule = self.kT_schedule

        # Initialize image
        device = next(self.parameters()).device
        img_shape = self.img_shape
        x = torch.randn((1, 1, *img_shape), device=device)

        # Loop over cycles
        for _ in range(num_cycles):

            # Increase temperature
            for kT in kT_schedule:
                x = x + torch.randn_like(x, device=x.device) * kT
                x = self(x)

            # Decrease temperature
            for kT in kT_schedule[-1::-1]:
                x = x + torch.randn_like(x, device=x.device) * kT
                x = self(x)

        # Return the result
        return x


# Test the model
if __name__ == '__main__':

    # Create a generator
    generator = Generator()

    # Create a random input
    x = torch.randn(1, 1, 64, 64)

    # Generate an image
    out = generator(x)

    # Dream a new image
    out = generator.dream()

    # Done!
    print("Done!")

