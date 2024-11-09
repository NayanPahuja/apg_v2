import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
from network import GeneratorAquaPixGan
import numpy as np
# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model
model = GeneratorAquaPixGan()
model.load_state_dict(torch.load('generator_70.pth', map_location=DEVICE))
model.eval()
model.to(DEVICE)

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Transform for output images
def denormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(tensor.device)
    return torch.clamp((tensor * std + mean) * 255, 0, 255)

def enhance_image(input_image):
    # Preprocess the input image
    input_tensor = transform(input_image).unsqueeze(0).to(DEVICE)

    # Generate the enhanced image
    with torch.no_grad():
        enhanced_tensor = model(input_tensor)

    # Postprocess the output
    enhanced_tensor = denormalize(enhanced_tensor[0].cpu())
    enhanced_image = Image.fromarray(enhanced_tensor.permute(1, 2, 0).numpy().astype(np.uint8))

    return enhanced_image