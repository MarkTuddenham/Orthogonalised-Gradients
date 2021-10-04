from torch import nn
import torch.nn.functional as F


class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.fc1_size = 32 * 4 * 4
        self.fc1 = nn.Linear(self.fc1_size, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(-1, self.fc1_size)
        x = self.fc1(x)
        return x


class BasicCNN_IR(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.fc1_size = 32 * 4 * 4
        self.fc1 = nn.Linear(self.fc1_size, 10)

        self.IR = []

    def forward(self, x):
        self.IR = []
        x = F.relu(self.bn1(self.conv1(x)))
        self.IR.append(x.clone().detach()[:10, :].cpu())
        x = F.relu(self.bn2(self.conv2(x)))
        self.IR.append(x.clone().detach()[:10, :].cpu())
        x = F.relu(self.bn3(self.conv3(x)))
        self.IR.append(x.clone().detach()[:10, :].cpu())

        x = x.view(-1, self.fc1_size)
        x = self.fc1(x)
        return x
