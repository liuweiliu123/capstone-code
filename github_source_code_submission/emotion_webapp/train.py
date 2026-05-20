import argparse
import json
import ssl
import urllib.error
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train the facial emotion recognition model.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../emotion_dataset",
        help="Path to the dataset root containing train/ and test/ folders.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    return parser.parse_args()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def build_model(num_classes: int):
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except (urllib.error.URLError, ssl.SSLError, RuntimeError) as exc:
        print(f"Could not download pretrained ResNet-18 weights. Training from random initialization. Reason: {exc}")
        model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("The dataset directory must contain train/ and test/ subdirectories.")

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_accuracy = 0.0
    best_path = output_dir / "best_model.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            running_correct += (predictions == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_accuracy = running_correct / running_total
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_accuracy={train_accuracy:.4f} | "
            f"test_loss={test_loss:.4f} test_accuracy={test_accuracy:.4f}"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), best_path)

    with open(output_dir / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, ensure_ascii=True, indent=2)

    with open(output_dir / "history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=True, indent=2)

    print(f"Best test accuracy: {best_accuracy:.4f}")
    print(f"Saved best model to: {best_path}")


if __name__ == "__main__":
    main()

