import torch
from torch import nn


class Baseline5(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_players, num_features)
        batch_size, sequence_length, num_players, num_features = x.shape
        x = x.view(batch_size * num_players, sequence_length, num_features)
        x, _ = self.lstm(x)  # (batch_size * num_players, sequence_length, hidden_size)
        x = x[:, -1, :]  # (batch_size * num_players, hidden_size)
        x = x.view(batch_size, num_players, -1)  # (batch_size, num_players, hidden_size)
        x = torch.max(x, 1)[0]  # (batch_size, hidden_size)
        return self.fc(x)  # (batch_size, num_classes)
