import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset


class LandcoverDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(
            self.mask_dir, self.images[index].replace(".jpg", "_mask.png")
        )

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        mask = mask.long().unsqueeze(0)

        return image, mask


def test():
    dataset = LandcoverDataset(
        image_dir="data/landcover/splitted/images",
        mask_dir="data/landcover/splitted/masks",
    )
    assert dataset[0][0].shape == (512, 512, 3)
    assert dataset[0][1].shape == (512, 512)

    # display the first image and mask
    import matplotlib.pyplot as plt

    plt.imshow(dataset[0][0])
    plt.imshow(dataset[0][1], alpha=0.5)
    print(dataset[0][1])
    plt.show()

    assert dataset[1][0].shape == (512, 512, 3)
    assert dataset[1][1].shape == (512, 512)
    assert dataset[2][0].shape == (512, 512, 3)
    assert dataset[2][1].shape == (512, 512)


if __name__ == "__main__":
    test()
