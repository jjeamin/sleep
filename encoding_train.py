import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
from med import resnet18
from med.utils import make_weights_for_balanced_classes, Encoding_Dataset
from torch.utils.data import Dataset
from pathlib import Path


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
    DATA_PATH = Path("store/public_dataset")

    train_data_path = DATA_PATH / "Fpz-Cz_train_encoding"
    valid_data_path = DATA_PATH / "Fpz-Cz_valid_encoding"

    transformer = transforms.Compose([
            transforms.ToTensor()
    ])

    train_dataset = Encoding_Dataset(root_path=train_data_path,
                                     transform=transformer)
    valid_dataset = Encoding_Dataset(root_path=valid_data_path,
                                     transform=transformer)

    weights = make_weights_for_balanced_classes(train_dataset.data, len(train_dataset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    train_total = len(train_loader)
    valid_total = len(valid_loader)

    model = resnet18(num_classes=5).to(args.device)

    # ############################## test
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
    # ##############################

    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30], gamma=0.1)

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
            torch.save(model.state_dict(), './resnet18_encoding.pth')
            best_acc = valid_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="store/public_dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", default=50)
    parser.add_argument("--batch_size", default=256)
    parser.add_argument("--lr", default=0.01)
    args = parser.parse_args()

    main(args)