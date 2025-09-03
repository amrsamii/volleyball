import torch
from torch import nn


class Baseline4(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_players, num_features)
        x = torch.max(x, dim=2)[0]  # (batch_size, sequence_length, num_features)
        x, _ = self.lstm(x)  # (batch_size, sequence_length, hidden_size)
        x = x[:, -1, :]  # (batch_size, hidden_size)
        return self.fc(x)  # (batch_size, num_classes)
