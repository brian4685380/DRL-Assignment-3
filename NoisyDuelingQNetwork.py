import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
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

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Buffers for noise
        self.register_buffer('weight_epsilon', torch.zeros(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1. / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        # Factorized Gaussian noise
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))  # outer product
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingHead(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(DuelingHead, self).__init__()
        self.value_stream = nn.Sequential(
            NoisyLinear(feature_dim, 256),
            nn.ReLU(),
            NoisyLinear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(feature_dim, 256),
            nn.ReLU(),
            NoisyLinear(256, output_dim)
        )

    def forward(self, x):
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    
class NoisyDuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NoisyDuelingQNetwork, self).__init__()
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

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()