from torch import nn


class Baseline7(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers, batch_first=True)
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
        x, _ = self.lstm1(x)  # (batch_size * num_players, sequence_length, hidden_size)
        x = x.view(batch_size, num_players, sequence_length, -1)

        # x = torch.max(x, 1)[0]  # (batch_size, sequence_length, hidden_size)

        # Apply adaptive max pool on num_players dimension
        x = x.permute(0, 2, 3, 1)  # (batch_size, sequence_length, hidden_size1, num_players)
        x = x.contiguous()
        x = x.view(batch_size * sequence_length, -1, num_players)
        x = self.adaptive_max_pool(x)  # (batch_size * sequence_length, hidden_size1, 1)
        x = x.squeeze()  # (batch_size * sequence_length, hidden_size1)
        x = x.view(batch_size, sequence_length, -1)

        x, _ = self.lstm2(x)  # (batch_size, sequence_length, hidden_size2)
        x = x[:, -1, :]  # (batch_size, hidden_size2)
        return self.fc(x)  # (batch_size, num_classes)
