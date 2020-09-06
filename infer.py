#!/usr/bin/env python
# -*- coding: utf-8 -*-

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
# import numpy as np
import pandas as pd
import os

workers = 0 if os.name == 'nt' else 4

# Determine if an nvidia GPU is available.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Define MTCNN module.
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    # keep_all=True,
    device=device
)

# Define Inception Resnet V1 module.
# noinspection PyUnresolvedReferences
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def collate_fn(ix):
    return ix[0]


# Define a dataset and data loader.
dataset = datasets.ImageFolder('./facenet_pytorch/data/test_images')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

# Perform MTCNN facial detection.
aligned = []
names = []
for x, y in loader:
    # noinspection PyTypeChecker
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

# Calculate image embeddings
aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

# Print distance matrix for classes.
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
print(pd.DataFrame(dists, columns=names, index=names))
