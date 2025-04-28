import torch
from torch import nn
import torch.nn.functional as F
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        return x

class DuelingHead(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(DuelingHead, self).__init__()
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    
class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingQNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor(input_dim[0]).to(self.device)

        dummy_input = torch.zeros(1, *input_dim).to(self.device)
        with torch.no_grad():
            feature_size = self.extractor(dummy_input).view(1, -1).size(1)

        self.head = DuelingHead(feature_size, output_dim).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x