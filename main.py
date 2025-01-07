import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data.volleyball_dataset import VolleyballDataset
from models.image_classifier import ImageClassifier
from utils import plot_confusion_matrix, plot_loss_accuracy

root_dir = "/media/amr/Extra/ML & DL/DL/projects/volleyball/dataset/"
train_dir = os.path.join(root_dir, "train")
test_dir = os.path.join(root_dir, "test")
validation_dir = os.path.join(root_dir, "validation")

device = "cuda" if torch.cuda.is_available() else "cpu"


batch_size = 64
# Any size that doesn't exceed ImageNet size
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

train_dataset = VolleyballDataset(train_dir, transform=transform)
train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True)

validation_dataset = VolleyballDataset(validation_dir, transform=transform)
validation_data_loader = DataLoader(validation_dataset, batch_size, shuffle=False)

model = ImageClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)


def train():
    model.train()

    train_loss = 0
    total_predictions = 0
    correct_predictions = 0

    for images, labels in train_data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        predicted = torch.argmax(outputs, dim=1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

        all_predictions.extend(predicted.cpu())
        all_labels.extend(labels.cpu())

    accuracy = 100 * correct_predictions / total_predictions
    train_loss /= len(train_data_loader)
    return accuracy, train_loss


def evaluate():
    model.eval()

    val_loss = 0
    val_correct_predictions = 0
    val_total_predictions = 0

    with torch.no_grad():
        for val_images, val_labels in validation_data_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_loss += criterion(val_outputs, val_labels).item()

            val_predicted = torch.argmax(val_outputs, dim=1)
            val_correct_predictions += (val_predicted == val_labels).sum().item()
            val_total_predictions += val_labels.size(0)

            val_all_predictions.extend(val_predicted.cpu())
            val_all_labels.extend(val_labels.cpu())

    val_accuracy = 100 * val_correct_predictions / val_total_predictions
    val_loss /= len(validation_data_loader)
    return val_accuracy, val_loss


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

    accuracy, train_loss = train()
    train_accuracies.append(accuracy)
    train_losses.append(train_loss)

    val_accuracy, val_loss = evaluate()
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
plot_confusion_matrix(all_labels, all_predictions, val_all_labels, val_all_predictions, "b1")
plot_loss_accuracy(train_losses, train_accuracies, val_losses, val_accuracies, "b1")
