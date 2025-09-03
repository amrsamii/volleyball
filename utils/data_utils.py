import os
from collections import defaultdict

import cv2

splits = {
    "train": [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
    "val": [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
    "test": [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47],
}


class Box:
    def __init__(self, data: list[str]):
        self.player_id, self.x1, self.y1, self.x2, self.y2 = map(int, data[:5])
        self.frame_id = data[5]
        self.action = data[-1]


class ClipAnnotation:
    """Each clip has 9 annotated frames."""

    def __init__(self, clip_id: str, frame_annotations: dict[str, list[Box]]):
        self.clip_id = clip_id
        self.frame_annotations = frame_annotations


class VideoAnnotation:
    def __init__(self, video_id: str, clip_activities: dict[str, str], clip_annotations: dict[str, ClipAnnotation]):
        self.video_id = video_id
        self.clip_activities = clip_activities
        self.clip_annotations = clip_annotations


def _load_clip_annotation(clip_id: str, tracking_annotations_path: str) -> ClipAnnotation:
    player_boxes = defaultdict(list)
    with open(tracking_annotations_path) as f:
        for line in f:
            box = Box(line.split())
            if box.player_id > 11:
                break
            player_boxes[box.player_id].append(box)

    frame_annotations = defaultdict(list)
    for boxes in player_boxes.values():
        for box in boxes[6:15]:
            frame_annotations[box.frame_id].append(box)

    for boxes in frame_annotations.values():
        boxes.sort(key=lambda box: (box.x1, box.x2))

    return ClipAnnotation(clip_id, frame_annotations)


def _load_clip_activities(annotation_path: str) -> dict[str, str]:
    clip_activities = {}

    with open(annotation_path) as f:
        for line in f:
            data = line.split()
            clip_id = data[0].split(".")[0]
            clip_activities[clip_id] = data[1]

    return clip_activities


def load_annotations(videos_dir: str, annotations_dir: str) -> dict[str, VideoAnnotation]:
    annotations = {}

    for video_id in sorted(os.listdir(videos_dir)):
        video_dir = os.path.join(videos_dir, video_id)
        clip_activities = _load_clip_activities(os.path.join(video_dir, "annotations.txt"))
        clip_annotations = {}
        for clip_id in os.listdir(video_dir):
            clip_dir = os.path.join(annotations_dir, video_id, clip_id)
            if not os.path.isdir(clip_dir):
                continue
            annotation_path = os.path.join(clip_dir, f"{clip_id}.txt")
            clip_annotation = _load_clip_annotation(clip_id, annotation_path)
            clip_annotations[clip_id] = clip_annotation

        annotations[video_id] = VideoAnnotation(video_id, clip_activities, clip_annotations)

    return annotations


def visualize_clip(videos_dir: str, video_id: int, clip_id: int, annotations: dict[str, VideoAnnotation]):
    annotation = annotations[str(video_id)]
    for frame_id, boxes in annotation.clip_annotations[str(clip_id)].frame_annotations.items():
        image_path = os.path.join(videos_dir, str(video_id), str(clip_id), f"{frame_id}.jpg")
        image = cv2.imread(image_path)
        for box in boxes:
            cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                box.action,
                (box.x1, box.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow(f"Video {video_id} Clip {clip_id}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
