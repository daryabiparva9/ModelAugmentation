import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Data loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 256
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten image
        return self.linear(x)  # CrossEntropyLoss applies softmax


# Training the logregression model
def train_logregression():
    logistic_model = LogisticRegressionModel().to(device)
    # Here I don't use reduction='sum' since for the original 
    # model I don't care about performance too much
    # A good enough model will do
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(logistic_model.parameters(), lr=0.001)
    # Training
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0
        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = logistic_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}')

    # Test phase
    logistic_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = logistic_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Base Logistic Regression Accuracy: {correct / total}')
    # Should give around 91-92% accuracy
    torch.save(logistic_model.state_dict(), 'models/logregression.pth')


if __name__ == "__main__":
    train_logregression()
