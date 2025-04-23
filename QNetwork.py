import torch
from torch import nn

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


class FullyConnected(nn.Module):
    def __init__(self, in_features, output_dim):
        super(FullyConnected, self).__init__()
        self.linear1 = nn.Linear(in_features, 512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.extractor = FeatureExtractor(input_dim[0]).to(self.device)

        # Compute feature size dynamically
        dummy_input = torch.zeros(1, *input_dim).to(self.device)
        with torch.no_grad():
            feature_size = self.extractor(dummy_input).view(1, -1).size(1)

        self.head = FullyConnected(feature_size, output_dim).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x