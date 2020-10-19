import os
import shutil
import random
import glob

channel = 'Pz-Oz'

DATASET_PATH = f'../store/public_dataset/{channel}'

use_label = ['Wake', 'N1', 'N2', 'N3', 'REM']

branch = [f'{channel}_train', f'{channel}_valid', f'{channel}_test']

print(os.listdir(DATASET_PATH))

for b in branch:
    for u in use_label:
        path = os.path.join(DATASET_PATH, b, u)

        if not os.path.exists(path):
            os.makedirs(path)

for u in use_label:
    load_paths = glob.glob(f"{os.path.join(DATASET_PATH, u)}/*")

    num_data = len(load_paths)

    split = int(num_data * 0.7)
    split2 = int(num_data * 0.9)

    train_paths = load_paths[:split]
    valid_paths = load_paths[split:split2]
    test_paths = load_paths[split2:]

    random.shuffle(load_paths)

    train_root_path = os.path.join(DATASET_PATH, branch[0], u)
    valid_root_path = os.path.join(DATASET_PATH, branch[1], u)
    test_root_path = os.path.join(DATASET_PATH, branch[2], u)

    for i, load_file_path in enumerate(train_paths):
        file_name = str(i) + ".png"

        save_file_path = os.path.join(train_root_path, file_name)

        shutil.copy(load_file_path, save_file_path)

    for i, load_file_path in enumerate(valid_paths):
        file_name = str(i) + ".png"

        save_file_path = os.path.join(valid_root_path, file_name)

        shutil.copy(load_file_path, save_file_path)

    for i, load_file_path in enumerate(test_paths):
        file_name = str(i) + ".png"

        save_file_path = os.path.join(test_root_path, file_name)

        shutil.copy(load_file_path, save_file_path)