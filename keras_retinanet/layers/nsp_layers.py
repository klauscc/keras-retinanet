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
import tensorflow as tf

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

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class TopRoiAligns(keras.layers.Layer):
    """Implements Selective ROIAlign. merge multi-level ROI regoins of feature maps"""

    def __init__(self,
                 pool_shape=[7, 7],
                 max_proposals=10,
                 score_threshold=0.00,
                 **kwargs):
        """ class initializers

        Args:
            **kwargs: Other optional arguments defined in @keras.layers.Layer

        Kwargs:
            pool_shape: A list of rank 2, representing [height, width] of the output pooled regions. Usually [7,7]
            max_proposals: Max number of proposals
            score_threshold: A float. The nms_classification greater than the score_threshold is taken into consideration.
        """
        super(TopRoiAligns, self).__init__(**kwargs)

        self._pool_shape = pool_shape
        self._max_proposals = max_proposals
        self._score_threshold = score_threshold

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs: a list of tensors: [features, boxes, nms_classification].
                features is a list of feature, i.e. [P3, P4, P5, P6].
                boxes is the normalized detection boxes
                nms_classification is the classification result after nms
            **kwargs: other args

        Returns: a list of feature. shape: [1, max_proposals, pool_shape[0], pool_shape[1], depth]. 1 is batch size (only 1 is supported). depth is feature (i.e. P3) channel.

        """
        # TODO: support batch_size > 1
        feature, boxes, nms_classification = inputs

        boxes = boxes[0]  # shape [n, 4]
        nms_classification = nms_classification[0]  # shape : [n,c]

        indices = tf.where(
            nms_classification > self._score_threshold)  # shape [num_true, 2]
        indices = tf.cast(indices, tf.int32)
        scores = tf.gather_nd(nms_classification, indices)  # shape [num_true]
        # shape [_max_proposals*2]
        # select twice the _max_proposals to ensure final selection can get more than _max_proposals
        k = tf.where(
            tf.shape(scores)[0] < self._max_proposals * 2,
            tf.shape(scores)[0], self._max_proposals * 2)
        max_n_scores, max_n_score_indices = tf.nn.top_k(scores, k=k)

        selected_indices = tf.gather(
            indices, max_n_score_indices)  # shape [_max_proposals*2, 2]
        # boxes_indices = tf.gather(
        # selected_indices, 0, axis=1)  # shape [_max_proposals*2]
        boxes_indices = selected_indices[:, 0]

        # remove duplicate indices
        boxes_unique_indices, _ = tf.unique(boxes_indices)
        boxes_unique_indices = boxes_unique_indices[:self._max_proposals]

        selected_boxes = tf.gather(
            boxes, boxes_unique_indices)  # shape [_max_proposals, 4]

        # convert [x1, y1, x2, y2] to [y1, x1, y2, x2]
        x1, y1, x2, y2 = tf.split(selected_boxes, 4, axis=1)
        selected_boxes = tf.concat([y1, x1, y2, x2], axis=1)
        boxes_ind = tf.zeros_like(boxes_unique_indices, dtype=tf.int32)

        # do roi pooling along each level of feature
        pooled = []
        paddings = tf.stack(
            [[0, self._max_proposals - tf.shape(boxes_unique_indices)[0]],
             [0, 0], [0, 0], [0, 0]])

        #stop gradient on selected boxes
        selected_boxes = tf.stop_gradient(selected_boxes)
        boxes_ind = tf.stop_gradient(boxes_ind)

        level_pooled = tf.image.crop_and_resize(
            feature, selected_boxes, boxes_ind, self._pool_shape
        )  # shape [n_selected, pool_height, pool_width, depth]

        #append n_selected to _max_proposals
        level_pooled = tf.pad(level_pooled, paddings)
        level_pooled = K.expand_dims(level_pooled, 0)
        return level_pooled

        # pooled.append(level_pooled)

        # roi_features = tf.stack(
        # pooled, axis=-1
        # )  # shape: [_max_proposals, pool_height, pool_width, depth, n_level]
        # if self._pool_type == 'MEAN':
        # reduce_func = tf.reduce_mean
        # else:
        # reduce_func = tf.reduce_max

        # fusion_roi_features = reduce_func(roi_features, axis=-1)
        # fusion_roi_features = tf.expand_dims(
        # fusion_roi_features,
        # 0)  # shape: [1, _max_proposals, pool_height, pool_width, depth]
        # return fusion_roi_features

    def compute_output_shape(self, input_shape):
        feature_shape = input_shape[0]
        depth = feature_shape[-1]
        output_shape = (None, self._max_proposals, self._pool_shape[0],
                        self._pool_shape[1], depth)
        return output_shape

    def get_config(self):
        config = super(TopRoiAligns, self).get_config()
        config.update({
            'pool_shape': self._pool_shape,
            'max_proposals': self._max_proposals,
            'score_threshold': self._score_threshold
        })


class SelectiveROI(keras.layers.Layer):
    """Docstring for  SelectiveROI. """

    def __init__(self, pool_type='MAX', **kwargs):
        """TODO: to be defined1. """
        super(SelectiveROI, self).__init__(**kwargs)
        self._pool_type = pool_type

    def call(self, inputs, **kwargs):
        """TODO: Docstring for call.

        Args:
            arg1 (TODO): TODO

        Returns: TODO

        """
        roi_features = tf.stack(
            inputs, axis=-1
        )  # shape: [batch_size,_max_proposals, pool_height, pool_width, depth, n_level]
        if self._pool_type == 'MEAN':
            reduce_func = tf.reduce_mean
        else:
            reduce_func = tf.reduce_max
        fusion_roi_features = reduce_func(
            roi_features, axis=-1
        )  # shape: [batch_size, _max_proposals, pool_height, pool_width, depth]
        return fusion_roi_features

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(SelectiveROI, self).get_config()
        config.update({
            'pool_type': self._pool_type,
        })
