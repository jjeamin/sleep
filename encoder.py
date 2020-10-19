import argparse
import torch
import os
import torchvision.transforms as transforms
from copy import deepcopy
from med import SegNet, SegNetv2
from med.utils import testing
from pathlib import Path
from tqdm import tqdm
from torchvision.datasets import ImageFolder


def train(model, train_loader, optimizer, device="cuda"):
    model.train()

    total = len(train_loader)
    train_loss = 0

    for images, _ in tqdm(train_loader, total=total):
        optimizer.zero_grad()

        images = images.to(device)

        true_images = deepcopy(images)
        pred_images = model(images)

        loss = torch.sqrt((pred_images - true_images).pow(2).mean())
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    return train_loss


def valid(model, valid_loader, device="cuda"):
    model.eval()

    total = len(valid_loader)
    valid_loss = 0

    for images, _ in tqdm(valid_loader, total=total):
        images = images.to(device)

        true_images = deepcopy(images)
        pred_images = model(images)

        loss = torch.sqrt((pred_images - true_images).pow(2).mean())

        valid_loss += loss.item()

    return valid_loss


def main(args):
    DATA_PATH = Path("store/public_dataset/Fpz-Cz")
    image_size = (128, 1024)

    total_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    total_dataset = ImageFolder(root=DATA_PATH, transform=total_transforms)
    total_length = len(total_dataset)
    split = [round(total_length * 0.7), round(total_length * 0.15), round(total_length * 0.15) - 1]

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(total_dataset, split)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    train_total = len(train_loader)
    valid_total = len(valid_loader)

    model = SegNet().to("cuda")
    model.load_state_dict(torch.load('./autoencoder_Fpz_Cz.pth'))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 80], gamma=0.1)

    minimum_loss = 1000

    for e in range(0, args.epochs):
        testing(model, test_loader, e)

        train_loss = train(model, train_loader, optimizer, device=args.device)
        train_loss = train_loss / train_total * args.batch_size

        valid_loss = valid(model, valid_loader, device=args.device)
        valid_loss = valid_loss / valid_total * args.batch_size

        scheduler.step()
        print(f"[EPOCH : {args.epochs} / {e}] || [TRAIN LOSS : {train_loss}] || [VALID LOSS : {valid_loss}]")

        if minimum_loss > valid_loss:
            torch.save(model.state_dict(), args.save_path)
            minimum_loss = valid_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8)
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--lr", default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_path", default="./autoencoder_fpz_cz.pth")
    args = parser.parse_args()

    main(args)