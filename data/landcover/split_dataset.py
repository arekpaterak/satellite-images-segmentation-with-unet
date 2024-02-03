import os
from sklearn.model_selection import train_test_split
import shutil

# Ścieżka do folderu z danymi
DATA_DIR = "data/landcover"

SOURCE_DIR = os.path.join(DATA_DIR, "splitted")

# Ścieżki do folderów images i masks
images_folder = os.path.join(SOURCE_DIR, "images")
masks_folder = os.path.join(SOURCE_DIR, "masks")

# Pobierz listę plików w folderze images
image_files = os.listdir(images_folder)

# Podział danych na train, val, test
train_and_val_files, test_files = train_test_split(
    image_files, test_size=0.1, random_state=42
)
train_files, val_files = train_test_split(
    train_and_val_files, test_size=0.2, random_state=42
)

# Ścieżki do folderów train, val, test
train_folder = os.path.join(DATA_DIR, "train")
val_folder = os.path.join(DATA_DIR, "val")
test_folder = os.path.join(DATA_DIR, "test")

# Utwórz foldery, jeśli nie istnieją
for folder in [train_folder, val_folder, test_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

    images_subfolder = os.path.join(folder, "images")
    if not os.path.exists(images_subfolder):
        os.makedirs(images_subfolder)

    masks_subfolder = os.path.join(folder, "masks")
    if not os.path.exists(masks_subfolder):
        os.makedirs(masks_subfolder)


# Kopiuj pliki do odpowiednich folderów
def copy_files(file_list, source_folder, destination_folder):
    for file in file_list:
        source_path = os.path.join(source_folder, "images", file)
        destination_path = os.path.join(destination_folder, "images", file)
        shutil.copy(source_path, destination_path)

        source_mask_path = os.path.join(
            source_folder, "masks", file.replace(".jpg", "_mask.png")
        )
        destination_mask_path = os.path.join(
            destination_folder, "masks", file.replace(".jpg", "_mask.png")
        )
        shutil.copy(source_mask_path, destination_mask_path)


copy_files(train_files, SOURCE_DIR, train_folder)
copy_files(val_files, SOURCE_DIR, val_folder)
copy_files(test_files, SOURCE_DIR, test_folder)
