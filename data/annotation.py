class PlayerAnnotation:
    def __init__(self, action: str, x: int, y: int, w: int, h: int):
        self.action = action
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self):
        return f"Action: {self.action}, X: {self.x}, Y: {self.y}, W: {self.w}, H: {self.h}"


class Annotation:
    def __init__(self, frame_id: str, video_id: str, group_activity: str, player_annotations: list[PlayerAnnotation]):
        self.frame_id = frame_id
        self.video_id = video_id
        self.group_activity = group_activity
        self.player_annotations = player_annotations

    def __str__(self):
        return (
            f"Frame ID: {self.frame_id}, Video ID: {self.video_id}, Group Activity: {self.group_activity}, "
            f"Player Annotations: {list(map(str, self.player_annotations))}"
        )
