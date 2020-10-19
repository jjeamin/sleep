import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False
from tqdm import tqdm
from med import resnet18
from med.utils import make_weights_for_balanced_classes
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from pathlib import Path
from timm import models

torch.backends.cudnn.benchmark = True


def train(model, train_loader, optimizer, criterion, device="cuda"):
    model.train()

    total = len(train_loader)
    train_correct = 0
    train_loss = 0

    for data, labels in tqdm(train_loader, total=total):
        optimizer.zero_grad()

        data = data.float().to(device)
        labels = labels.to(device)

        pred = model(data)
        _, predicted = torch.max(pred, 1)

        train_correct += (predicted == labels).sum().item()

        loss = criterion(pred, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    return train_correct, train_loss


def valid(model, valid_loader, criterion, device="cuda"):
    model.eval()

    total = len(valid_loader)
    valid_correct = 0
    valid_loss = 0

    for data, labels in tqdm(valid_loader, total=total):
        data = data.float().to(device)
        labels = labels.to(device)

        pred = model(data)
        _, predicted = torch.max(pred, 1)

        valid_correct += (predicted == labels).sum().item()

        loss = criterion(pred, labels)
        valid_loss += loss.item()

    return valid_correct, valid_loss


def main(args):
    DATA_PATH = Path(args.data_path)
    image_size = (128, 1024)

    train_data_path = DATA_PATH / f"{args.channel}_train"
    valid_data_path = DATA_PATH / f"{args.channel}_valid"

    train_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    valid_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolder(root=train_data_path, transform=train_transforms)
    valid_dataset = ImageFolder(root=valid_data_path, transform=valid_transforms)

    weights = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    train_total = len(train_loader)
    valid_total = len(valid_loader)

    model = resnet18(num_classes=5).to(args.device)
    # model = models.efficientnet_b3()
    # model.conv_stem = nn.Conv2d(1, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # model.classifier = nn.Linear(in_features=1536, out_features=5, bias=True)

    model = model.to(args.device)

    ############################## test
    # data_iter = iter(train_loader)
    # data, labels = data_iter.next()
    #
    # data = data.float().to("cuda")
    #
    # print(data.shape)
    #
    # output = model(data)
    #
    # print(output.shape)
    ##############################

    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 80], gamma=0.1)

    best_acc = 0

    for e in range(0, args.epochs):
        train_correct, train_loss = train(model, train_loader, optimizer, criterion, device=args.device)
        train_acc = train_correct / (train_total * args.batch_size)
        train_loss = train_loss / (train_total * args.batch_size)

        valid_correct, valid_loss = valid(model, valid_loader, criterion, device=args.device)
        valid_acc = valid_correct / (valid_total * args.batch_size)
        valid_loss = valid_loss / (valid_total * args.batch_size)

        scheduler.step()
        print(f"[EPOCH : {args.epochs} / {e}] || [TRAIN ACC : {train_acc}] || [TRAIN LOSS : {train_loss}]"
              f"|| [VALID ACC : {valid_acc}] || [VALID LOSS : {valid_loss}]")

        if best_acc < valid_acc:
            torch.save({'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       args.save_path)
            best_acc = valid_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="store/public_dataset")
    parser.add_argument("--save_path", default="./resnet18_pz.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--batch_size", default=256)
    parser.add_argument("--lr", default=0.01)
    parser.add_argument("--channel", default='Pz-Oz')
    args = parser.parse_args()

    main(args)