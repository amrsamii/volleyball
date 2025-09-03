import torch
from torch import nn


class Baseline8(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_players, num_features)
        batch_size, sequence_length, num_players, num_features = x.shape
        x = x.view(batch_size * num_players, sequence_length, num_features)
        x, _ = self.lstm1(x)  # (batch_size * num_players, sequence_length, hidden_size1)
        x = x.view(
            batch_size, num_players, sequence_length, -1
        )  # (batch_size, num_players, sequence_length, hidden_size1)

        team1 = x[:, :6, :, :]  # (batch_size, 6, sequence_length, hidden_size1)
        team2 = x[:, 6:, :, :]  # (batch_size, 6, sequence_length, hidden_size1)
        team1 = torch.max(team1, dim=1)[0]
        team2 = torch.max(team2, dim=1)[0]
        # team1 = team1.contiguous().view(batch_size * sequence_length, -1, 6)
        # team2 = team2.contiguous().view(batch_size * sequence_length, -1, 6)
        # team1 = self.adaptive_max_pool(team1)  # (batch_size * sequence_length, hidden_size1, 1)
        # team2 = self.adaptive_max_pool(team2)  # (batch_size * sequence_length, hidden_size1, 1)
        # team1 = team1.squeeze()
        # team2 = team2.squeeze()
        # team1 = team1.view(batch_size, sequence_length, -1)
        # team2 = team2.view(batch_size, sequence_length, -1)

        x = torch.cat((team1, team2), dim=2)  # (batch_size, sequence_length, hidden_size1 * 2)
        x, _ = self.lstm2(x)  # (batch_size, sequence_length, hidden_size2)
        x = x[:, -1, :]  # (batch_size, hidden_size2)
        return self.fc(x)  # (batch_size, num_classes)
