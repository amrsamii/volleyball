import os

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_confusion_matrix(all_labels, all_predictions, val_all_labels, val_all_predictions, display_labels, plot_name):
    cm = confusion_matrix(all_labels, all_predictions)
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(ax=ax).figure_.savefig(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "confusion_matrix", f"{plot_name}_training")
    )
    plt.close(fig)

    val_cm = confusion_matrix(val_all_labels, val_all_predictions)
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(val_cm, display_labels=display_labels).plot(ax=ax).figure_.savefig(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "confusion_matrix", f"{plot_name}_validation")
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
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loss_accuracy", f"{plot_name}.png"))


def train_epoch(
    model, train_data_loader, criterion, optimizer, all_predictions, all_labels
) -> tuple[float, float, float]:
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
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    return accuracy, train_loss, f1


def evaluate_epoch(
    model, validation_data_loader, criterion, val_all_predictions, val_all_labels
) -> tuple[float, float, float]:
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
    val_f1 = f1_score(val_all_labels, val_all_predictions, average="weighted")
    return val_accuracy, val_loss, val_f1


def train(model, criterion, optimizer, train_data_loader, validation_data_loader, epochs, display_labels, name, logger):
    logger.info("Start training...")

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

        accuracy, train_loss, train_f1 = train_epoch(
            model, train_data_loader, criterion, optimizer, all_predictions, all_labels
        )
        train_accuracies.append(accuracy)
        train_losses.append(train_loss)

        val_accuracy, val_loss, val_f1 = evaluate_epoch(
            model, validation_data_loader, criterion, val_all_predictions, val_all_labels
        )
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        logger.info(
            f"Epoch {epoch + 1}, Loss: {train_loss}, Accuracy: {accuracy:.2f}%, F1 Score: {train_f1:.2f}, "
            f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy:.2f}%, Validation F1 Score: {val_f1:.2f}"
        )

    logger.info("Training completed.")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trained_models", f"{name}_weights.pth")
    torch.save(model.state_dict(), path)
    logger.info(f"Model weights saved to {path}")
    plot_confusion_matrix(all_labels, all_predictions, val_all_labels, val_all_predictions, display_labels, name)
    plot_loss_accuracy(train_losses, train_accuracies, val_losses, val_accuracies, name)


def evaluate(model, criterion, test_data_loader, logger):
    test_accuracy, test_loss, test_f1 = evaluate_epoch(model, test_data_loader, criterion, [], [])

    logger.info(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy:.2f}%, Test F1 Score: {test_f1:.2f}")
