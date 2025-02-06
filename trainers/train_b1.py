import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from constants import group_activities
from datasets.group_dataset import GroupDataset
from models.group_classifier import GroupClassifier
from utils.data_utils import load_annotations
from utils.logger import get_logger
from utils.train_utils import train

logger = get_logger("b1.log")

videos_dir = "/media/amr/Extra/ML & DL/DL/projects/volleyball/videos/"
annotations_dir = "/media/amr/Extra/ML & DL/DL/projects/volleyball/tracking_annotation/"
annotations = load_annotations(videos_dir, annotations_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 16

# Any size that doesn't exceed ImageNet size
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

logger.info("Loading Training dataset...")
train_dataset = GroupDataset(videos_dir, annotations, "train", transform=transform)
train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True)
logger.info(f"Training dataset size: {len(train_dataset)}")

logger.info("Loading Validation dataset...")
validation_dataset = GroupDataset(videos_dir, annotations, "val", transform=transform)
validation_data_loader = DataLoader(validation_dataset, batch_size, shuffle=False)
logger.info(f"Validation dataset size: {len(validation_dataset)}")

model = GroupClassifier(len(group_activities)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
train(
    model,
    criterion,
    optimizer,
    train_data_loader,
    validation_data_loader,
    10,
    group_activities.keys(),
    "b1",
    logger,
)
