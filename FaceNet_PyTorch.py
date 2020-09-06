#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from Utils import UtilsCommon as utils
from .facenet_pytorch import MTCNN
from .facenet_pytorch import InceptionResnetV1
from .facenet_pytorch import extract_face
from .facenet_pytorch import fixed_image_standardization


_this_folder_    = os.path.dirname(os.path.abspath(__file__))
_this_basename_  = os.path.splitext(os.path.basename(__file__))[0]
_this_algorithm_ = 'FaceNet_PyTorch'


class Handler:

    def __init__(self, ini=None, logger=utils.get_stdout_logger()):

        self.ini = ini
        self.logger = logger

        self.description = None
        self.version = None
        self.device = None

        self.img_size = None
        self.margin = None
        self.min_face_size = None
        self.thresholds = None
        self.factor = None
        self.post_process_ = True

        self.pretrained = None
        self.classify_ = False
        self.num_classes = None
        self.dropout_prob = None
        self.embed_resol = None

        self.mtcnn = None
        self.resnet = None

        if ini:
            self.init_ini(ini)

    def init_ini(self, ini):

        self.description   =       ini['description']
        self.version       =       ini['version']
        self.device        =       ini['device']

        self.init_ini_mtcnn(ini['MTCNN'])

        self.pretrained    =       ini['pretrained']
        self.classify_     =  bool(ini['classify_'])
        self.num_classes   =       ini['num_classes']
        self.dropout_prob  = float(ini['dropout_prob'])
        self.embed_resol   =   int(ini['embed_resol'])

        try:
            # noinspection PyUnresolvedReferences
            self.device = torch.device(self.device.lower())
        except all:
            self.device = torch.device('cpu')

        self.num_classes = None if self.num_classes == 'None' else int(self.num_classes)

        self.logger.info(" # {} : initializing...".format(_this_algorithm_))

        self.init_net()

    def init_ini_mtcnn(self, ini):
        self.img_size      =   int(ini['img_size'])
        self.margin        =   int(ini['margin'])
        self.min_face_size =   int(ini['min_face_size'])
        self.thresholds    =  eval(ini['thresholds'])
        self.factor        = float(ini['factor'])
        self.post_process_ =  bool(ini['post_process_'])

    def init_net(self):

        self.mtcnn = MTCNN(image_size=self.img_size,
                           margin=self.margin,
                           min_face_size=self.min_face_size,
                           thresholds=self.thresholds,
                           factor=self.factor,
                           post_process=self.post_process_,
                           select_largest=False,
                           keep_all=True,
                           device=self.device)
        # noinspection PyUnresolvedReferences
        self.resnet = InceptionResnetV1(pretrained=self.pretrained,
                                        classify=self.classify_,
                                        num_classes=self.num_classes,
                                        dropout_prob=self.dropout_prob,
                                        device=self.device).eval().to(self.device)
        self.logger.info(" # {} : loading {} to {} device ...".
                         format(_this_algorithm_, self.pretrained, self.device))

    def run(self, img):
        # Detect faces
        with torch.no_grad():
            batch_boxes, batch_probs = self.mtcnn.detect(img)
        if batch_boxes is None:
            return [], [], [], []

        face_img_arr = []
        for i, box in enumerate(batch_boxes):
            face_img = extract_face(img, box, self.img_size, self.margin, save_path=None)
            if self.post_process_:
                face_img = fixed_image_standardization(face_img)
            face_img_arr.append(face_img)
        face_img_arr = torch.stack(face_img_arr)

        # faces, probs = self.mtcnn(img, save_path=face_fname, return_prob=True)

        face_vectors = []
        face_indexes = []
        face_errors = []
        resolution = np.power(10, self.embed_resol)
        if self.classify_:
            face_recog_probs = self.resnet(face_img_arr.to(self.device)).detach().cpu()
            for idx, arr in enumerate(face_recog_probs):
                abs_arr = abs(arr)
                face_indexes.append(int(abs_arr.argmin()))
                face_errors.append(utils.trunc(abs_arr[face_indexes[-1]], 0.001))
                embed_vec = [int(x * resolution) / resolution for x in self.resnet.vec[idx]]
                face_vectors.append(embed_vec)
        else:
            # Calculate embedding (un-squeeze to add batch dimension)
            face_vectors = self.resnet(face_img_arr)

        return batch_boxes, face_vectors, face_indexes, face_errors
