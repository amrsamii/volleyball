import os
import pickle

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from constants import group_activities, num_features
from datasets.features_dataset import FeaturesDataset
from models.baseline5 import Baseline5
from utils.logger import get_logger
from utils.train_utils import evaluate, train

logger = get_logger("b5.log")

batch_size = 64

features_dir = "/media/amr/Extra/ML & DL/DL/projects/volleyball/features/"

with open(os.path.join(features_dir, "train_features.pkl"), "rb") as f:
    train_features = pickle.load(f)
with open(os.path.join(features_dir, "val_features.pkl"), "rb") as f:
    val_features = pickle.load(f)
with open(os.path.join(features_dir, "test_features.pkl"), "rb") as f:
    test_features = pickle.load(f)

logger.info("Loading Training dataset...")
train_dataset = FeaturesDataset(train_features)
train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True)
logger.info(f"Training dataset size: {len(train_dataset)}")

logger.info("Loading Validation dataset...")
validation_dataset = FeaturesDataset(val_features)
validation_data_loader = DataLoader(validation_dataset, batch_size, shuffle=False)
logger.info(f"Validation dataset size: {len(validation_dataset)}")

logger.info("Loading Test dataset...")
test_dataset = FeaturesDataset(test_features)
test_data_loader = DataLoader(test_dataset, batch_size, shuffle=False)
logger.info(f"Test dataset size: {len(test_dataset)}")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Baseline5(num_features, 1024, 1, len(group_activities)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
train(
    model,
    criterion,
    optimizer,
    train_data_loader,
    validation_data_loader,
    30,
    group_activities.keys(),
    "b5",
    logger,
)

evaluate(model, criterion, test_data_loader, logger)
