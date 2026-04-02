import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

MODEL_PATH = "trained_resnet18_binary.pth"
IMG_PATH = "detec.png"
IMG_SIZE = 224
THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

image = Image.open(IMG_PATH).convert("RGB")
original_image = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
image_tensor = transform(image).unsqueeze(0).to(device)

activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

target_layer = model.layer4[-1]
forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_full_backward_hook(backward_hook)

output = model(image_tensor)
prob_ai = torch.sigmoid(output).item()

if prob_ai >= THRESHOLD:
    predicted_class = "AI"
    confidence = prob_ai
    target_score = output
else:
    predicted_class = "REAL"
    confidence = 1 - prob_ai
    target_score = -output 

print(f"Raw sigmoid probability (AI): {prob_ai:.4f}")
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence * 100:.2f}%")

model.zero_grad()
target_score.backward()

pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

activations_detached = activations.detach().squeeze(0)

for i in range(activations_detached.shape[0]):
    activations_detached[i] *= pooled_gradients[i]

heatmap = torch.mean(activations_detached, dim=0).cpu().numpy()
heatmap = np.maximum(heatmap, 0)

if np.max(heatmap) != 0:
    heatmap /= np.max(heatmap)

heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
heatmap_uint8 = np.uint8(255 * heatmap_resized)

heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)

cv2.imwrite("gradcam_heatmap.jpg", heatmap_color)
cv2.imwrite("gradcam_overlay.jpg", overlay)

print("Saved: gradcam_heatmap.jpg")
print("Saved: gradcam_overlay.jpg")

high_attention = np.mean(heatmap_resized > 0.6) * 100
very_high_attention = np.mean(heatmap_resized > 0.8) * 100

print("\nExplanation:")
if predicted_class == "AI":
    print("- The model predicted this image as AI-generated.")
    print("- Grad-CAM shows which regions influenced the AI decision most.")
    if very_high_attention > 15:
        print("- The model focused strongly on a concentrated set of regions, suggesting suspicious local artifacts.")
    elif high_attention > 30:
        print("- The model used several medium-to-high attention regions across the image.")
    else:
        print("- The model's attention is relatively spread out, so the AI signal is weaker and less localized.")
else:
    print("- The model predicted this image as REAL.")
    print("- Grad-CAM shows which regions supported the REAL decision most.")
    if very_high_attention > 15:
        print("- The model relied strongly on specific realistic-looking regions.")
    elif high_attention > 30:
        print("- The model found multiple supportive natural-looking regions.")
    else:
        print("- The model's REAL decision is based on distributed evidence rather than one strong suspicious area.")

forward_handle.remove()
backward_handle.remove()