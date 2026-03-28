import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "trained_vgg16_binary.pth"
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class_names = ['AI', 'REAL']

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = models.vgg16(weights=None)

in_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features, 1)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
def predict_image(file):
    image = Image.open(file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()

    if prob >= 0.65:
        predicted_class = class_names[1]
        confidence = prob
    else:
        predicted_class = class_names[0]
        confidence = 1 - prob
    print(f"Raw sigmoid probability: {prob:.4f}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")
    return predicted_class, confidence