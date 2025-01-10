import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data.volleyball_dataset import VolleyballDataset, group_activities
from models.image_classifier import ImageClassifier
from utils import evaluate_epoch, plot_confusion_matrix, plot_loss_accuracy, train_epoch

root_dir = "/media/amr/Extra/ML & DL/DL/projects/volleyball/dataset/"
train_dir = os.path.join(root_dir, "train")
test_dir = os.path.join(root_dir, "test")
validation_dir = os.path.join(root_dir, "validation")

device = "cuda" if torch.cuda.is_available() else "cpu"


batch_size = 64
# Any size that doesn't exceed ImageNet size
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = VolleyballDataset(train_dir, transform=transform)
train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True)

validation_dataset = VolleyballDataset(validation_dir, transform=transform)
validation_data_loader = DataLoader(validation_dataset, batch_size, shuffle=False)

model = ImageClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

epochs = 20
patience = 5
previous_val_accuracy = 0
epochs_without_improvement = 0

train_accuracies = []
train_losses = []
val_accuracies = []
val_losses = []
all_labels = []
all_predictions = []
val_all_predictions = []
val_all_labels = []

for epoch in range(epochs):
    all_labels = []
    all_predictions = []
    val_all_predictions = []
    val_all_labels = []

    accuracy, train_loss = train_epoch(model, train_data_loader, criterion, optimizer, all_predictions, all_labels)
    train_accuracies.append(accuracy)
    train_losses.append(train_loss)

    val_accuracy, val_loss = evaluate_epoch(
        model, validation_data_loader, criterion, val_all_predictions, val_all_labels
    )
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch + 1}, Loss: {train_loss}, "
        f"Accuracy: {accuracy:.2f}%, Validation Loss: {val_loss}, "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    if val_accuracy - previous_val_accuracy >= 1:
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping due to no improvement in val accuracy.")
        break

    previous_val_accuracy = val_accuracy

torch.save(model.state_dict(), "trained_models/b1_weights.pth")
plot_confusion_matrix(all_labels, all_predictions, val_all_labels, val_all_predictions, group_activities.keys(), "b1")
plot_loss_accuracy(train_losses, train_accuracies, val_losses, val_accuracies, "b1")
