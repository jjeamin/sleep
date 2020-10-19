import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Encoding_Dataset(Dataset):
    def __init__(self,
                 root_path,
                 transform=None):
        self.root_path = root_path
        self.data = []
        self.transform = transform
        self.classes = os.listdir(root_path)

        for c in self.classes:
            class_path = root_path / c
            for data_path in os.listdir(class_path):
                self.data.append((class_path / data_path, self.classes.index(c)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = np.load(self.data[i][0])
        y = self.data[i][1]

        return x, y

def make_weights_for_balanced_classes(data, nclasses):
    count = [0] * nclasses
    for item in data:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(data)
    for idx, val in enumerate(data):
        weight[idx] = weight_per_class[val[1]]
    return weight


def testing(model, loader, e, device="cuda"):
    data_iter = iter(loader)
    true_images, _ = data_iter.next()

    true_images = true_images.to(device)
    pred_images = model(true_images).to(device)

    fig = plt.figure(figsize=(8, 8))

    rows, columns = 4, 2

    for i, (t, p) in enumerate(zip(true_images[:4], pred_images[:4])):
        t = t[0].cpu().detach().numpy()
        p = p[0].cpu().detach().numpy()

        fig.add_subplot(rows, columns, i * 2 + 1)
        plt.imshow(t, cmap=cm.gray)
        fig.add_subplot(rows, columns, (i + 1) * 2)
        plt.imshow(p, cmap=cm.gray)

    plt.show()
    plt.savefig(f'outputs/{e}.png', dpi=300)