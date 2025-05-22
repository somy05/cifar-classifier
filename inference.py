import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose
import os
import sys

from train import CNNBase

def evaluate_model():
    device = torch.device('cpu')
    print(f"Using {device} device")

    batch_size = 64  
    model_path = "cnnmodel.pth"
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    test_transform = Compose([
        ToTensor(),
        Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])   #values taken from: https://github.com/kuangliu/pytorch-cifar/issues/19                            
    ])

    test_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=test_transform,
        )
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=6, persistent_workers=True)

    model = CNNBase(num_classes=len(classes)).to(device)

    if os.path.isfile(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"model weight found at: {model_path}")
                print(f"Model trained for {checkpoint['epoch'] + 1} epochs")
                print(f"Final training loss was: {checkpoint['loss']:.4f}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return
    else:
        print(f"Model file not found at {model_path}")
        return

    model.eval() 
    correct = 0
    total = 0
    

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the {total} test images: {accuracy:.2f} %')
    sys.exit(0)

if __name__ == '__main__':
    evaluate_model()