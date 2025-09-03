import torch
from torch import nn


class Baseline3B(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_players, num_features)
        # Average pool over 9 frames summarizing the information across the sequence
        x = torch.mean(x, dim=1)  # (batch_size, num_players, num_features)
        # Max pool over 12 players
        x = torch.max(x, dim=1)[0]  # (batch_size, num_features)
        return self.fc(x)  # (batch_size, num_classes)
