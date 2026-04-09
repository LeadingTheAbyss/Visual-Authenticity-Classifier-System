import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

model_path = "trained_resnet18_binary.pth"
sz = 224
cutoff = 0.6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize((sz, sz)), transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
model = models.resnet18(weights = None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(model_path, map_location = device))
model = model.to(device)
model.eval()

def predict_image(file):
    image = Image.open(file).convert("RGB")
    original = np.array(image.resize((sz, sz)))
    tensor = transform(image).unsqueeze(0).to(device)
    act = None
    grad = None

    def fhook(module, inp, out):
        nonlocal act
        act = out

    def bhook(module, grad_inp, grad_out):
        nonlocal grad
        grad = grad_out[0]

    target = model.layer4[-1]
    front = target.register_forward_hook(fhook)
    back = target.register_full_backward_hook(bhook)
    prob_ai = torch.sigmoid(model(tensor)).item()

    if prob_ai >= cutoff:
        pred = "AI"
        confidence = prob_ai
        target = model(tensor)
    else:
        pred = "REAL"
        confidence = 1 - prob_ai
        target = -model(tensor)

    model.zero_grad()
    target.backward()
    pooled_grad = torch.mean(grad, dim = [0, 2, 3])
    act = act.detach().squeeze(0)

    for i in range(act.shape[0]):
        act[i] *= pooled_grad[i]

    heatmap = torch.mean(act, dim = 0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    h_r = cv2.resize(heatmap, (sz, sz))
    col = cv2.applyColorMap(np.uint8(255 * h_r), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(original, cv2.COLOR_RGB2BGR), 0.6, col, 0.4, 0)
    cv2.imwrite("gradcam_heatmap.jpg", col)
    cv2.imwrite("gradcam_overlay.jpg", overlay)
    hi = np.mean(h_r > 0.6) * 100
    vhi = np.mean(h_r > 0.8) * 100

    if pred == "AI":
        if vhi > 15:
            reason = "Model has focused mainly on a concentrated set of regions which suggests suspicious local artifacts."
        elif hi > 30:
            reason = "Model used several medium to high attention regions from the image."
        else:
            reason = "Model's attention is spread out so therefore the AI signal is weaker and less localized."
    else:
        if vhi > 15:
            reason = "Model strongly relies on specific realistic looking regions."
        elif hi > 30:
            reason = "Model found multiple regions which are supportive and natural looking."
        else:
            reason = "Model's decision is based on distributed evidence rather than one strong suspicious area."
        
    front.remove()
    back.remove()
    return pred, confidence, reason
