import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim

torch.set_num_threads(8)
torch.set_num_interop_threads(8)

def main():

    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10


    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=6, persistent_workers=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size,num_workers=6, persistent_workers=True)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    torch.cuda.is_available = lambda : False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    class CNNBase(nn.Module):
        def __init__(self, num_classes=10):
            super(CNNBase, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 8 * 8, num_classes)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            return x
        
    model = CNNBase(num_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch, (X, y) in enumerate(train_dataloader):
            # Compute prediction and loss
            predict = model(X)
            loss = loss_fn(predict, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    avg_loss = total_loss / len(train_dataloader)       
    print(f"Training done, loss: {avg_loss:.4f}")   

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X, y in test_dataloader:
            output = model(X)
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
        accuracy = float(correct) / float(total) * 100
        print(f"Got {correct}/{total} with accuracy {accuracy:.2f}%!")
    model.train()

if __name__ == "__main__":
    main()
