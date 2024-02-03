import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

from torchinfo import summary

from model import UNet

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 3

PIN_MEMORY = False
LOAD_MODEL = False

TRAIN_IMG_DIR = "data/landcover/train/images/"
TRAIN_MASK_DIR = "data/landcover/train/masks/"
VAL_IMG_DIR = "data/landcover/val/images/"
VAL_MASK_DIR = "data/landcover/val/masks/"

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


def train_step(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)

            targets = targets.long()
            targets = targets.squeeze(1)

            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print("Loss: ", loss.item())

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNet(in_channels=3, n_classes=5).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    summary(model, (BATCH_SIZE, 3, 512, 512))

    print("Loading data...")
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        print("Loading model...")
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # for batch_idx, (data, targets) in enumerate(train_loader):
    #     print(data.shape)
    #     print(targets.shape)
    #     print(torch.unique(targets))
    #     break

    # return

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        train_step(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )


if __name__ == "__main__":
    main()
