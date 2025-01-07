from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from data.volleyball_dataset import group_activities


def plot_confusion_matrix(all_labels, all_predictions, val_all_labels, val_all_predictions, plot_name):
    cm = confusion_matrix(all_labels, all_predictions)
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(cm, display_labels=group_activities.keys()).plot(ax=ax).figure_.savefig(
        f"confusion_matrix/{plot_name}_training"
    )
    plt.close(fig)

    val_cm = confusion_matrix(val_all_labels, val_all_predictions)
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(val_cm, display_labels=group_activities.keys()).plot(ax=ax).figure_.savefig(
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
