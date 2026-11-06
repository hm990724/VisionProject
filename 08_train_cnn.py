"""Train ResNet18 or MobileNetV3-Small on existing processed image splits.
Example: python 08_train_cnn.py --task eyes --model resnet18
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def make_model(name, pretrained=True):
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    else:
        raise ValueError(name)
    return model


def get_transforms(training):
    ops = []
    if training:
        # Light augmentation, applied only to training images.
        ops += [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(5)]
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = total_correct = n = 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            total_correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, total_correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, choices=['eyes', 'yawn'])
    p.add_argument('--model', required=True, choices=['resnet18', 'mobilenet_v3_small'])
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--data', default='data/processed')
    a = p.parse_args()
    if a.epochs <= 0 or a.batch <= 0:
        p.error('epochs and batch must be positive')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available() else ('cpu' if a.device == 'auto' else a.device))
    if device.type == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is not available')
    folder = ROOT / a.data / a.task
    train_ds = datasets.ImageFolder(folder / 'train', transform=get_transforms(True))
    val_ds = datasets.ImageFolder(folder / 'val', transform=get_transforms(False))
    if train_ds.class_to_idx != val_ds.class_to_idx or len(train_ds.classes) != 2:
        raise RuntimeError('Train/val classes do not match or are not binary')
    train_loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers)
    val_loader = DataLoader(val_ds, batch_size=a.batch, shuffle=False, num_workers=a.workers)
    out = ROOT / 'runs' / f'{a.task}_{a.model}'
    if (out / 'best.pt').exists():
        raise SystemExit(f'Existing model would be overwritten: {out / "best.pt"}. Back it up or move it first.')
    out.mkdir(parents=True, exist_ok=True)
    print('Classes:', train_ds.class_to_idx, '| device:', device)
    print('Training uses preprocessed 224x224 images and ImageNet normalization.')
    model = make_model(a.model, pretrained=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, stale = -1.0, 0
    with (out / 'history.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
        for epoch in range(1, a.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optim)
            val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc]); f.flush()
            print(f'Epoch {epoch:02d} | train loss={train_loss:.4f} acc={train_acc:.2%} | val loss={val_loss:.4f} acc={val_acc:.2%}')
            if val_acc > best_acc + 1e-8:
                best_acc, stale = val_acc, 0
                torch.save({'model': model.state_dict(), 'architecture': a.model, 'classes': train_ds.classes,
                            'best_epoch': epoch, 'val_accuracy': val_acc}, out / 'best.pt')
            else:
                stale += 1
                if stale >= a.patience:
                    print(f'Early stopping at epoch {epoch}, best validation accuracy={best_acc:.2%}')
                    break
    print('Saved:', out / 'best.pt')


if __name__ == '__main__':
    main()
