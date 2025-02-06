import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from constants import group_activities
from utils.data_utils import VideoAnnotation, splits


class GroupDataset(Dataset):
    def __init__(self, videos_dir: str, annotations: dict[str, VideoAnnotation], split: str, transform=None):
        self.data = []

        for video_id in splits[split]:
            annotation = annotations[str(video_id)]
            for clip_id in annotation.clip_activities:
                clip_activity = annotation.clip_activities[clip_id]
                for frame_id in annotation.clip_annotations[clip_id].frame_annotations:
                    image_path = os.path.join(videos_dir, str(video_id), clip_id, f"{frame_id}.jpg")
                    self.data.append((image_path, clip_activity))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        example = self.data[idx]
        image = Image.open(example[0])
        # No need for one-hot encoding because we are using CrossEntropyLoss which expects class indices
        label = torch.tensor(group_activities[example[1]])

        if self.transform:
            image = self.transform(image)

        return image, label
