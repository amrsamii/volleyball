import torch
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_confusion_matrix(all_labels, all_predictions, val_all_labels, val_all_predictions, display_labels, plot_name):
    cm = confusion_matrix(all_labels, all_predictions)
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(ax=ax).figure_.savefig(
        f"confusion_matrix/{plot_name}_training"
    )
    plt.close(fig)

    val_cm = confusion_matrix(val_all_labels, val_all_predictions)
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(val_cm, display_labels=display_labels).plot(ax=ax).figure_.savefig(
        f"confusion_matrix/{plot_name}_validation"
    )
    plt.close(fig)


def plot_loss_accuracy(train_losses, train_accuracies, val_losses, val_accuracies, plot_name):
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
    plt.savefig(f"loss_accuracy/{plot_name}.png")


def train_epoch(model, train_data_loader, criterion, optimizer, all_predictions, all_labels) -> tuple[float, float]:
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


def evaluate_epoch(
    model, validation_data_loader, criterion, val_all_predictions, val_all_labels
) -> tuple[float, float]:
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
