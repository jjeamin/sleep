import os
from PIL import Image
from pathlib import Path


DATA_PATH = Path("store/ae_dataset")

valid_path = DATA_PATH / "test"

classes = os.listdir(valid_path)

for c in classes:
    for i in os.listdir(valid_path / c):
        img_path = valid_path / c / i

        img = Image.open(img_path)

        print(img.size)