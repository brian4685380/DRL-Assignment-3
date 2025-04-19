import torch
from torch import nn
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.extract_feature = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    @property
    def feature_size(self):
        x = self.extract_feature(
            torch.zeros(1, *self.input_dim)
        )
        return x.view(1, -1).size(1)
        
    def forward(self, x):
        x = self.extract_feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        