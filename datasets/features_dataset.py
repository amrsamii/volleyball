import torch
from torch.utils.data import Dataset

from constants import group_activities


class FeaturesDataset(Dataset):
    def __init__(self, features):
        self.data = []
        for video_id in features:
            for clip_id in features[video_id]:
                clip_activity = features[video_id][clip_id]["label"]
                clip_features = features[video_id][clip_id]["features"]
                self.data.append((clip_features, clip_activity))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        clip_features = self.data[idx][0]
        label = torch.tensor(group_activities[self.data[idx][1]])
        return clip_features, label
