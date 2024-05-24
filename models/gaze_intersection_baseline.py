import torch
class GazeIntersection():

    def __init__(self):
        super().__init__()

    def forward(self, x):
        img_mask = x[:, 0, :, :]
        gaze_heatmap = x[:, 1, :, :]
        intersection_mask = img_mask * gaze_heatmap
        return intersection_mask