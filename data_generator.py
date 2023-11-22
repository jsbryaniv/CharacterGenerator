
# Import libraries
import torch
import unicodedata
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, RandomAffine, GaussianBlur

        
# Chinese character dataset
class ChineseCharacters(Dataset):
    """Chinese character dataset."""
    
    def __init__(self, image_size=(64, 64), transform=True):
        super().__init__()

        # Generate a list of all the Chinese characters
        start = int("4E00", 16)
        end = int("9FFF", 16)
        characters = [chr(i) for i in range(start, end+1) if unicodedata.category(chr(i)).startswith('Lo')]
        
        # Set the parameters
        self.image_size = image_size
        self.transform = transform
        self.characters = characters
        self.num_characters = len(characters)
        self.font_path = "fonts/NotoSansSC-Regular.otf"

    def __len__(self):
        return self.num_characters

    def __getitem__(self, idx):
        
        # Get the character
        character = self.characters[idx]

        # Get the image
        image = self.print_character(character)

        # Distort the image
        if self.transform:
            image = RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.8, 1),
                shear=(0.1, 0.1),
                fill=1
            )(image)
            image = GaussianBlur(kernel_size=9, sigma=.01*max(self.image_size))(image)

        # Finalize the image
        image -= image.min()
        if image.max() > 0:
            image /= image.max()

        return image

    def _get_font_size(self, text, font_ratio):

        # Calculate the maximum font size based on the image size
        max_font_size = int(max(self.image_size) * font_ratio)

        # Load the font file
        font = ImageFont.truetype(self.font_path, size=max_font_size)

        # Calculate the size of the text with the maximum font size
        text_bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), text, font)
        text_size = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # Calculate the font size that fits within the image
        font_size = int(font_ratio * max(self.image_size) / max(text_size) * max_font_size)
        return font_size

    def print_character(self, text, font_ratio=0.8):

        # Create an image
        image = Image.new('L', self.image_size, color=255)

        # Get the font
        font_size = self._get_font_size(text, font_ratio)
        font = ImageFont.truetype(self.font_path, size=font_size)

        # Get the text size and location
        text_bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), text, font)
        text_size = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        ascent = int(font.getmetrics()[1] * font_size / text_size[1])
        x = (self.image_size[0] - text_size[0]) // 2
        y = (self.image_size[1] - text_size[1]) // 2 - ascent

        # Draw the text
        draw = ImageDraw.Draw(image)
        draw.text((x, y), text, font=font, fill=0)

        # Convert the image to a tensor
        image = ToTensor()(image)

        # Return the image
        return image


# Test the dataset
if __name__ == "__main__":

    # Create a dataset
    dataset = ChineseCharacters()

    # Get a random character
    fig = plt.gcf()
    fig.clf()
    plt.ion()
    plt.show()
    for i in range(100):
        image = dataset[i]
        plt.clf()
        plt.imshow(image[0, :, :].detach().numpy(), cmap='gray')
        plt.pause(.1)

    print("Done.")