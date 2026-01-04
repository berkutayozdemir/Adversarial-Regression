import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, z_dim=10):
        super(Generator, self).__init__()
        # Input: Features (X) + Noise (Z)
        # Architecture: 256 -> 128 -> 64 -> 1
        # All ReLU activations
        self.net = nn.Sequential(
            nn.Linear(input_dim + z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, z):
        # Concatenate x and z along the feature dimension
        inp = torch.cat([x, z], dim=1)
        return self.net(inp)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        # Input: Features (X) + Target (Y)
        # Architecture: 256 -> 256 -> 256 -> 1
        # Leaky ReLU activations
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 256),
            nn.LeakyReLU(0.2), # Standard slope for Leaky ReLU
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() # Output probability [0, 1]
        )

    def forward(self, x, y):
        # Concatenate x and y
        inp = torch.cat([x, y], dim=1)
        return self.net(inp)
