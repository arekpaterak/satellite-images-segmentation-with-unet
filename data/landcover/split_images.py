import glob
import os

import cv2

TARGET_SIZE = 512

DATA_DIR = os.path.join("data/landcover")

IMAGES_DIR = os.path.join(DATA_DIR, "raw", "images")
MASKS_DIR = os.path.join(DATA_DIR, "raw", "masks")

img_paths = glob.glob(os.path.join(IMAGES_DIR, "*.tif"))
mask_paths = glob.glob(os.path.join(MASKS_DIR, "*.tif"))

img_paths.sort()
mask_paths.sort()

OUTPUT_IMAGES_DIR = os.path.join(DATA_DIR, "splitted", "images")
OUTPUT_MASKS_DIR = os.path.join(DATA_DIR, "splitted", "masks")

os.makedirs(OUTPUT_IMAGES_DIR)
os.makedirs(OUTPUT_MASKS_DIR)

for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
    img_filename = os.path.splitext(os.path.basename(img_path))[0]
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    assert img_filename == mask_filename and img.shape[:2] == mask.shape[:2]

    k = 0
    for y in range(0, img.shape[0], TARGET_SIZE):
        for x in range(0, img.shape[1], TARGET_SIZE):
            img_tile = img[y : y + TARGET_SIZE, x : x + TARGET_SIZE]
            mask_tile = mask[y : y + TARGET_SIZE, x : x + TARGET_SIZE]

            if img_tile.shape[0] == TARGET_SIZE and img_tile.shape[1] == TARGET_SIZE:
                out_img_path = os.path.join(
                    OUTPUT_IMAGES_DIR, f"{img_filename}_{k}.jpg"
                )
                cv2.imwrite(out_img_path, img_tile)

                out_mask_path = os.path.join(
                    OUTPUT_MASKS_DIR, f"{mask_filename}_{k}_mask.png"
                )
                cv2.imwrite(out_mask_path, mask_tile)

            k += 1

    print(f"Processed {img_filename} {i + 1}/{len(img_paths)}")
