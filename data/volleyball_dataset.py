import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.utils import load_data

group_activities = {
    "r_spike": 0,
    "r_set": 1,
    "r-pass": 2,
    "r_winpoint": 3,
    "l-spike": 4,
    "l_set": 5,
    "l-pass": 6,
    "l_winpoint": 7,
}


class VolleyballDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.annotations = load_data(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int):
        annotation = self.annotations[idx]
        image_path = os.path.join(self.root_dir, annotation.video_id, annotation.frame_id, f"{annotation.frame_id}.jpg")
        image = Image.open(image_path)
        # No need for one-hot encoding because we are using CrossEntropyLoss which expects class indices
        label = torch.tensor(group_activities[annotation.group_activity])

        if self.transform:
            image = self.transform(image)

        return image, label
