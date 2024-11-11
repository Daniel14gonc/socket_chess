import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ChessConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ChessResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class ChessNetwork(nn.Module):
    def __init__(self, num_res_blocks=19, num_channels=256, num_moves=4672):
        super().__init__()

        # Input layer - Corregido: 12 canales de entrada, num_channels de salida
        self.conv_input = ChessConvBlock(12, num_channels)  # Cambiado de (num_channels, 12) a (12, num_channels)

        # Residual tower
        self.res_tower = nn.Sequential(
            *[ChessResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # El resto permanece igual...
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, num_moves)

        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x shape viene como: (batch_size, 8, 8, 12)
        # Necesitamos convertirlo a: (batch_size, 12, 8, 8)
        x = x.permute(0, 3, 1, 2)  # Reorganiza las dimensiones
        
        x = self.conv_input(x)
        x = self.res_tower(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.reshape(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.reshape(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value