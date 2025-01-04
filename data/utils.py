import os

import cv2

from data.annotation import Annotation, PlayerAnnotation


def load_data(root_dir: str):
    annotations = []

    for video_id in sorted(os.listdir(root_dir)):
        video_dir = os.path.join(root_dir, video_id)
        with open(os.path.join(video_dir, "annotations.txt")) as f:
            for line in f:
                data = line.split()
                frame_id = data[0].split(".")[0]
                frame_activity = data[1]

                player_annotations = []
                for i in range(2, len(data), 5):
                    x, y, w, h = map(int, data[i : i + 4])
                    action = data[i + 4]
                    player_annotations.append(PlayerAnnotation(action, x, y, w, h))

                annotations.append(Annotation(frame_id, video_id, frame_activity, player_annotations))

    return annotations


def visualize_example(root_dir: str, idx: int, annotations: list[Annotation]):
    annotation = annotations[idx]
    frame_path = os.path.join(root_dir, annotation.video_id, annotation.frame_id, f"{annotation.frame_id}.jpg")
    img = cv2.imread(frame_path)

    for player_annotation in annotation.player_annotations:
        cv2.rectangle(
            img,
            (player_annotation.x, player_annotation.y),
            (player_annotation.x + player_annotation.w, player_annotation.y + player_annotation.h),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img,
            player_annotation.action,
            (player_annotation.x, player_annotation.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imshow(f"Example {idx}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
