import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Load the PyTorch model
model = torch.load('models/saved_yolo_glare_model3.pt')
model.eval()  # Set model to evaluation mode