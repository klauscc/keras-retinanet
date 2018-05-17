#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: PyramidROIAlign.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/05/14
#   description:
#
#================================================================

import keras
from .. import backend
import keras.backend as K

import numpy as np


class NormalizeBoxes(keras.layers.Layer):
    """normalize the values of boxes to [0,1]"""

    def __init__(self, **kwargs):
        super(NormalizeBoxes, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """TODO: Docstring for call.

        Args:
            inputs: list of [image, boxes]. image is the network input tensor, boxes is the proposals.
            **kwargs: 
        Returns: normalized boxes. each box (x1,y1,x2,y2)               is in range [0,1) 
        """
        image, boxes = inputs
        shape = keras.backend.cast(
            keras.backend.shape(image), keras.backend.floatx())
        x1 = boxes[:, :, 0] / shape[2]
        y1 = boxes[:, :, 1] / shape[1]
        x2 = boxes[:, :, 2] / shape[2]
        y2 = boxes[:, :, 3] / shape[1]
        return K.stack([x1, y1, x2, y2], axis=2)


class SelectiveROIAlign(keras.layers.Layer):
    """Implements Selective ROIAlign. merge multi-level ROI regoins of feature maps"""

    def __init__(self, pool_shape, max_proposals=10, **kwargs):
        """ class initializers

        Args:
            pool_shape: a list of rank 2, representing [height, width] of the output pooled regions. Usually [7,7]
            max_proposals: max number of proposals
            **kwargs: other optional arguments defined in @keras.layers.Layer
        """
        super(SelectiveROIAlign, self).__init__(**kwargs)

        self._pool_shape = pool_shape

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs: a list of tensors: [features, boxes, nms_classification]. 
                features is a list of feature, i.e. [P3, P4, P5, P6].
                boxes is the normalized detection boxes
                nms_classification is the classification result after nms
            **kwargs: other args

        Returns: a list of feature with pool_shape

        """
        pass
