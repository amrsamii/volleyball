import os
import pickle
from collections import defaultdict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from constants import actions, num_features
from models.baseline3_a import Baseline3A
from utils.data_utils import load_annotations, splits
from utils.logger import get_logger

logger = get_logger("extract_features.log")

device = "cuda" if torch.cuda.is_available() else "cpu"

videos_dir = "/media/amr/Extra/ML & DL/DL/projects/volleyball/videos/"
annotations_dir = "/media/amr/Extra/ML & DL/DL/projects/volleyball/tracking_annotation/"
features_dir = "/media/amr/Extra/ML & DL/DL/projects/volleyball/features/"

annotations = load_annotations(videos_dir, annotations_dir)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

features_model = Baseline3A(len(actions))
features_model.load_state_dict(torch.load("trained_models/b3_a_2_weights.pth", weights_only=True))
features_model = nn.Sequential(*list(features_model.children())[:-1]).to(device)
features_model.eval()

with torch.no_grad():
    for split in splits:
        logger.info(f"Extracting {split} features...")
        split_features = defaultdict(dict)

        for video_id in splits[split]:
            logger.info(f"Extracting video {video_id} features...")
            annotation = annotations[str(video_id)]
            for clip_id in annotation.clip_activities:
                clip_activity = annotation.clip_activities[clip_id]
                clip_features = []

                for frame_id, boxes in annotation.clip_annotations[clip_id].frame_annotations.items():
                    image_path = os.path.join(videos_dir, str(video_id), clip_id, f"{frame_id}.jpg")
                    image = Image.open(image_path)
                    cropped_images = []

                    for box in boxes:
                        cropped_image = image.crop(
                            (
                                box.x1,
                                box.y1,
                                box.x2,
                                box.y2,
                            )
                        )
                        cropped_images.append(transform(cropped_image))

                    cropped_images = torch.stack(cropped_images).to(device)
                    dnn_repr = features_model(cropped_images)
                    dnn_repr = dnn_repr.view(cropped_images.size(0), -1)

                    while dnn_repr.shape[0] < 12:
                        dnn_repr = torch.cat((dnn_repr, torch.zeros(1, num_features).to(device)))
                    clip_features.append(dnn_repr.cpu())

                clip_features = torch.stack(clip_features)
                split_features[video_id][clip_id] = {"features": clip_features, "label": clip_activity}

        pkl_file_path = os.path.join(features_dir, f"{split}_features_2.pkl")
        with open(pkl_file_path, "wb") as f:
            pickle.dump(split_features, f)

        logger.info(f"Saved {len(split_features)} {split} videos to {pkl_file_path}")
