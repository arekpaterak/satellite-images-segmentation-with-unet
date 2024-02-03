import gradio

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import cv2
import PIL
import matplotlib.pyplot as plt

from PIL import Image
import io

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import UNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(
    in_channels=3,
    n_classes=5,
    features=[64, 128, 256],
)

model.load_state_dict(torch.load("model.pth"))

model.to(DEVICE)
model.eval()

transform = A.Compose(
    [
        A.Resize(height=64, width=64),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)


def segment(image: np.ndarray):
    image = transform(image=image)["image"]

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)
        mask = torch.argmax(probs, dim=1)

    mask = mask[0].squeeze(0).cpu().detach().numpy()

    return mask


def plot_image(image):
    mask = segment(image)

    fig, ax = plt.subplots()

    cax = ax.imshow(mask, vmin=0, vmax=4, cmap="viridis")
    cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3, 4], orientation="horizontal")
    cbar.ax.set_xticklabels(["Other", "Building", "Woodland", "Water", "Road"])
    ax.axis("off")

    return fig


inputs = [gradio.Image(label="Input Image")]
outputs = [gradio.Plot(label="Segmentation")]

interface = gradio.Interface(
    fn=plot_image,
    inputs=inputs,
    outputs=outputs,
    allow_flagging=False,
)

if __name__ == "__main__":
    interface.launch()
