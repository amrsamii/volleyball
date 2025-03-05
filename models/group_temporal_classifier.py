from torch import nn


class GroupTemporalClassifier(nn.Module):
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
        # x shape (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # out shape (batch_size, sequence_length, hidden_size)
        out = out[:, -1, :]
        # out shape (batch_size, hidden_size)
        out = self.fc(out)
        # out shape (batch_size, num_classes)
        return out
