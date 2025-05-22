import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomCrop, Normalize, ColorJitter
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os

class CNNBase(nn.Module):
        def __init__(self, num_classes=10):
            super(CNNBase, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
            self.batchnorm1 = nn.BatchNorm2d(32)
            self.batchnorm2 = nn.BatchNorm2d(64)
            self.batchnorm3 = nn.BatchNorm2d(128)
            self.dropout = nn.Dropout(0.2)



        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.batchnorm1(x)
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.batchnorm2(x)
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.batchnorm3(x)
            x = self.pool(x)
            x = self.dropout(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            return x

def main():

    
    torch.set_num_threads(6)
    device = torch.device('cpu')
    print(f"Using {device}")
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 30
    model_path = "cnnmodel.pth"
    
    train_transform = Compose([
        RandomHorizontalFlip(p=0.5), 
        RandomCrop(size=32, padding = 4),   
        ToTensor(),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])       #values taken from: https://github.com/kuangliu/pytorch-cifar/issues/19                             
    ])


    training_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=train_transform,
        )
        


    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=6, persistent_workers=True)
   

    model = CNNBase(num_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    #add weight decay as regularisation term
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
    else:
        print("No existing model, training from scratch")
    
    model.train()
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for batch, (X, y) in enumerate(train_dataloader):
           
            X, y = X.to(device), y.to(device)
            predict = model(X)
            loss = loss_fn(predict, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predictedclasses = torch.argmax(predict, dim=1)
            total += y.size(0)
            correct += torch.sum((predictedclasses == y)).item()
        print(f"Epoch {epoch+1}/{num_epochs}, accuracy: {100*(correct/total):.2f}%")
   

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_path)

if __name__ == "__main__":
    main()