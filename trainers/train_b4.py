import os
import pickle

import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from constants import group_activities, num_features
from datasets.temporal_features_dataset import TemporalFeaturesDataset
from models.group_temporal_classifier import GroupTemporalClassifier
from utils.logger import get_logger
from utils.train_utils import train

logger = get_logger("b4.log")

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 64

features_dir = "/media/amr/Extra/ML & DL/DL/projects/volleyball/features/"

with open(os.path.join(features_dir, "train_features.pkl"), "rb") as f:
    train_features = pickle.load(f)
with open(os.path.join(features_dir, "val_features.pkl"), "rb") as f:
    val_features = pickle.load(f)

logger.info("Loading Training dataset...")
train_dataset = TemporalFeaturesDataset(train_features)
train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True)
logger.info(f"Training dataset size: {len(train_dataset)}")

logger.info("Loading Validation dataset...")
validation_dataset = TemporalFeaturesDataset(val_features)
validation_data_loader = DataLoader(validation_dataset, batch_size, shuffle=False)
logger.info(f"Validation dataset size: {len(validation_dataset)}")

classifier_model = GroupTemporalClassifier(num_features, 1024, 1, len(group_activities)).to(device)
criterion = CrossEntropyLoss()
optimizer = optim.AdamW(classifier_model.parameters(), lr=0.0001)

train(
    classifier_model,
    criterion,
    optimizer,
    train_data_loader,
    validation_data_loader,
    20,
    group_activities.keys(),
    "b4",
    logger,
)
