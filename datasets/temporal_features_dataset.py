import torch
from torch.utils.data import Dataset

from constants import group_activities, num_features


class TemporalFeaturesDataset(Dataset):
    def __init__(self, features):
        self.data = []
        for video_id in features:
            for clip_id in features[video_id]:
                clip_activity = features[video_id][clip_id]["label"]
                clip_features = features[video_id][clip_id]["features"]
                frame_features = torch.split(clip_features, num_features)
                frame_features = torch.stack(frame_features)
                self.data.append((frame_features, clip_activity))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        image_features = self.data[idx][0]
        label = torch.tensor(group_activities[self.data[idx][1]])
        return image_features, label
