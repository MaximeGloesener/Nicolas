# imports 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import *
from torchvision.datasets import CIFAR10

from utils.benchmark import benchmark
from tqdm.auto import tqdm

from registry import get_model

import os 
import re 

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# evaluation loop
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    verbose=True,
) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0
    loss = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False, disable=not verbose):
        # Move the data from CPU to GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Inference
        outputs = model(inputs)
        # Calculate loss
        loss += F.cross_entropy(outputs, targets, reduction="sum")
        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)
        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()
    return (num_correct / num_samples * 100).item(), (loss / num_samples).item()



# dataloader pour cifar10 et cifar100
NORMALIZE_DICT = {
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    'cifar100': dict( mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2761)),
    }
image_size = 32
transforms = {
    "train": Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(**NORMALIZE_DICT["cifar10"]),
    ]),
    "test": Compose([
        ToTensor(),
        Normalize(**NORMALIZE_DICT["cifar10"]),
    ]),
}
dataset = {}
for split in ["train", "test"]:
    dataset[split] = CIFAR10(root="data/cifar10", train=(split == "train"), download=True, transform=transforms[split])
dataloader = {}
for split in ['train', 'test']:
    dataloader[split] = DataLoader(dataset[split], batch_size=256, shuffle=(split == 'train'), num_workers=0, pin_memory=True)



models = ["resnet56", "vgg11_bn", "vgg16_bn", "repvgg_a0"]

# Analyse des modèles initiaux non prunés 
for model_name in models:
    print(f'----- Modèle = {model_name} -----')
    model = get_model(model_name, "cifar10").to(device)
    acc, loss = evaluate(model, dataloader["test"])
    print(f"Accuracy: {acc:.2f}%")
    print(f"Loss: {loss:.4f}")
    benchmark(model, torch.randn(1, 3, 32, 32).to(device))
    print("\n\n")

    pruned_path = 'results'
    for pruned_model in os.listdir(pruned_path):
        if model_name in pruned_model:
            method = re.search(r'cifar10_(.*?)\.pth', pruned_model).group(1)
            model = re.search(r'(.*?)_compression', pruned_model).group(1)
            compression_ratio = float(re.search(r'compression_(.*?)_', pruned_model).group(1))
            print(f'** Modèle = {model} **')
            print(f'** Méthode = {method} **')
            print(f'** Compression Ratio = {compression_ratio} **')
            model = torch.load(os.path.join(pruned_path, pruned_model)).to(device)
            acc, loss = evaluate(model, dataloader["test"])
            print(f"Accuracy: {acc:.2f}%")
            print(f"Loss: {loss:.4f}")
            benchmark(model, torch.randn(1, 3, 32, 32).to(device))
            print("\n\n")



