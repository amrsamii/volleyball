import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from constants import actions
from utils.data_utils import VideoAnnotation, splits


class PersonDataset(Dataset):
    def __init__(self, videos_dir: str, annotations: dict[str, VideoAnnotation], split: str, transform=None):
        self.data = []

        for video_id in splits[split]:
            annotation = annotations[str(video_id)]
            for clip_id in annotation.clip_activities:
                for frame_id, boxes in annotation.clip_annotations[clip_id].frame_annotations.items():
                    image_path = os.path.join(videos_dir, str(video_id), clip_id, f"{frame_id}.jpg")
                    image = Image.open(image_path)
                    for box in boxes:
                        cropped_image = image.crop((box.x1, box.y1, box.x2, box.y2))
                        self.data.append((cropped_image, box.action))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        image = self.data[idx][0]
        # No need for one-hot encoding because we are using CrossEntropyLoss which expects class indices
        label = torch.tensor(actions[self.data[idx][1]])

        if self.transform:
            image = self.transform(image)

        return image, label
