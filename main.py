import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data.volleyball_dataset import VolleyballDataset, group_activities
from models.image_classifier import ImageClassifier

root_dir = "/media/amr/Extra/ML & DL/DL/projects/volleyball/dataset/"
train_dir = os.path.join(root_dir, "train")
test_dir = os.path.join(root_dir, "test")
validation_dir = os.path.join(root_dir, "validation")

device = "cuda" if torch.cuda.is_available() else "cpu"


# Any size that doesn't exceed ImageNet size
train_dataset = VolleyballDataset(
    train_dir, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
)
train_data_loader = DataLoader(train_dataset, 64, shuffle=True)

validation_dataset = VolleyballDataset(
    validation_dir, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
)
validation_data_loader = DataLoader(validation_dataset, 64, shuffle=False)

model = ImageClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

epochs = 20
patience = 5
previous_accuracy = 0
epochs_without_improvement = 0

train_accuracies = []
train_losses = []
val_accuracies = []
val_losses = []

for epoch in range(epochs):
    train_loss = 0
    total_predictions = 0
    correct_predictions = 0
    all_labels = []
    all_predictions = []

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

    accuracy = correct_predictions / total_predictions
    train_accuracies.append(accuracy)

    train_loss /= len(train_data_loader)
    train_losses.append(train_loss)

    model.eval()

    val_loss = 0
    val_correct_predictions = 0
    val_total_predictions = 0
    val_all_predictions = []
    val_all_labels = []

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

    val_accuracy = val_correct_predictions / val_total_predictions
    val_accuracies.append(val_accuracy)

    val_loss /= len(validation_data_loader)
    val_losses.append(val_loss)

    model.train()

    print(
        f"Epoch {epoch + 1}, Loss: {train_loss}, "
        f"Accuracy: {accuracy * 100:.2f}%, Validation Loss: {val_loss}, "
        f"Validation Accuracy: {val_accuracy * 100:2f}%"
    )

    if (epoch + 1) % 10 == 0:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_accuracies": train_accuracies,
                "val_accuracies": val_accuracies,
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
            f"checkpoints/b1/checkpoint_epoch_{epoch + 1}.pth",
        )

        cm = confusion_matrix(all_labels, all_predictions)
        fig, ax = plt.subplots(figsize=(10, 10))
        ConfusionMatrixDisplay(cm, display_labels=group_activities.keys()).plot(ax=ax).figure_.savefig(
            f"confusion_matrix/b1/training_{epoch + 1}"
        )
        plt.close(fig)

        val_cm = confusion_matrix(val_all_labels, val_all_predictions)
        fig, ax = plt.subplots(figsize=(10, 10))
        ConfusionMatrixDisplay(val_cm, display_labels=group_activities.keys()).plot(ax=ax).figure_.savefig(
            f"confusion_matrix/b1/validation_{epoch + 1}"
        )
        plt.close(fig)

    if val_accuracy - previous_accuracy >= 0.01:
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping due to no improvement in val accuracy.")
        break

    previous_accuracy = val_accuracy

epochs_list = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))  # Create a new figure

plt.subplot(1, 2, 1)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.plot(epochs_list, train_accuracies, label="Training Accuracy")
plt.plot(epochs_list, val_accuracies, label="Validation Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.plot(epochs_list, train_losses, label="Training Loss")
plt.plot(epochs_list, val_losses, label="Validation Loss")
plt.legend()

plt.tight_layout()
plt.savefig("loss_accuracy/b1.png")
