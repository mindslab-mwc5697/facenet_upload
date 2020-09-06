#!/usr/bin/env python
# -*- coding: utf-8 -*-
from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
# from IPython import display


# Determine if an nvidia GPU is available.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Define MTCNN module.
mtcnn = MTCNN(keep_all=True, device=device)

# Get a sample video.
vid_cap = cv2.VideoCapture('./facenet_pytorch/examples/video.mp4')

# Run video through MTCNN.
i = 0
frames_tracked = []
while vid_cap.isOpened():
    success, frame = vid_cap.read()
    if not success:
        break
    frame = Image.fromarray(np.uint8(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).convert('RGB')
    print('\rTracking frame: {}'.format(i + 1), end='')
    i += 1

    # Detect faces.
    # noinspection PyUnresolvedReferences
    boxes, _ = mtcnn.detect(frame)

    # Draw faces.
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

    # Add to frame list
    frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
print('\nDone')

# Display detections.
"""
d = display.display(frames_tracked[0], display_id=True)
i = 1
try:
    while True:
        d.update(frames_tracked[i % len(frames_tracked)])
        i += 1
except KeyboardInterrupt:
    pass
"""

# Save tracked video.
dim = frames_tracked[0].size
# noinspection PyUnresolvedReferences
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
# noinspection PyUnresolvedReferences
video_tracked = cv2.VideoWriter('./facenet_pytorch/examples/video_tracked.mp4', fourcc, 25.0, dim)
for frame in frames_tracked:
    # noinspection PyUnresolvedReferences
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
video_tracked.release()
