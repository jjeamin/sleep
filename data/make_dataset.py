import pandas as pd
import os
import glob
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

DATASET_PATH = './ae_dataset'

branch = 'total'

use_label = ['Wake', 'N1', 'N2', 'N3', 'REM']

anns = glob.glob('./data/ann*')
images = glob.glob('./data/image*')

for u in use_label:
    path = os.path.join(DATASET_PATH, branch, u)

    if not os.path.exists(path):
        os.makedirs(path)

file_numbers = []
start_epoch = []

name_idx = [0, 0, 0, 0, 0]

for ann, image in zip(anns, images):
    df = pd.read_csv(ann)
    use_df = df[df['Event'].isin(use_label)]

    file_names = os.listdir(image)
    labels = use_df["Event"].values

    print(len(file_names), len(labels))
    print("==============================")

    for i, (label, epoch) in enumerate(zip(labels, use_df['Start Epoch'].values)):
        load_path = os.path.join(image, epoch + '.png')

        event_idx = use_label.index(label)

        # save_file_name = str(name_idx[event_idx]) + ".png"

        # save_path = os.path.join(DATASET_PATH, branch, label, save_file_name)

        save_path_1 = os.path.join(DATASET_PATH, branch, label, str(name_idx[event_idx]) + "_0.png")
        save_path_2 = os.path.join(DATASET_PATH, branch, label, str(name_idx[event_idx]) + "_1.png")
        save_path_3 = os.path.join(DATASET_PATH, branch, label, str(name_idx[event_idx]) + "_2.png")
        save_path_4 = os.path.join(DATASET_PATH, branch, label, str(name_idx[event_idx]) + "_3.png")

        img = Image.open(load_path)
        crop_img = img.crop((1, 157, img.width, 310))
        # crop_img = img.crop((1, 41.5, img.width, img.height-9))

        channel1 = crop_img.crop((0, 0, crop_img.width, 36.5))
        channel2 = crop_img.crop((0, 39, crop_img.width, 75))
        channel3 = crop_img.crop((0, 77.5, crop_img.width, 114.5))
        channel4 = crop_img.crop((0, 117, crop_img.width, crop_img.height - 1))

        channel1.save(save_path_1)
        channel2.save(save_path_2)
        channel3.save(save_path_3)
        channel4.save(save_path_4)

        # shutil.copy(load_path, save_path)

        name_idx[event_idx] += 1