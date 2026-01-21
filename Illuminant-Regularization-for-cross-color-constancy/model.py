import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class OneNet_PhysicsResidual(nn.Module):
    def __init__(self):
        super().__init__()

        self.residual = ResidualBlock(3)

        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=1)
        self.dropout = nn.Dropout(0.5)

        self.conv5 = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):

        res = self.residual(x)
        alpha = torch.sigmoid(self.alpha)
        x = (1 - alpha) * x + alpha * res

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.conv5(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), 3)
        x = F.normalize(x, p=2, dim=1)
        return x
