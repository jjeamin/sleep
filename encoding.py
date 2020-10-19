import torch
import os
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from med import SegNet, get_encoder
from pathlib import Path

branch = ["valid", 'test']

for b in branch:
    DATA_PATH = Path("store/public_dataset")
    total_data_path = DATA_PATH / f"Fpz-Cz_{b}"
    encoding_path = DATA_PATH / f"Fpz-Cz_{b}_encoding"

    classes = os.listdir(total_data_path)

    # make dir
    if not os.path.exists(encoding_path):
        os.mkdir(encoding_path)

        for c in classes:
            os.mkdir(encoding_path / c)

    image_size = (128, 1024)

    transformer = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor()
    ])

    model = SegNet().to("cuda")
    model.load_state_dict(torch.load("./autoencoder_Fpz_Cz.pth"))

    encoder = get_encoder(model)
    encoder.eval()

    for c in classes:
        class_data_path = total_data_path / c
        class_data_paths = os.listdir(class_data_path)

        encoding = []

        for n in range(0, len(class_data_paths)):
            total = []

            img_path = class_data_path / f"{n}.png"
            print(img_path)
            img = Image.open(img_path)
            tensor_img = transformer(img).unsqueeze(0).to("cuda")

            output = encoder(tensor_img) # 128 x 1024 -> 128 x 512
            output = output.reshape(1, 1024, -1).cpu().detach().numpy()

            np.save(encoding_path / c / f'{n}.npy', output)

