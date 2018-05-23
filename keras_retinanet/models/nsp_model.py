#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: nsp_model.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/05/18
#   description:
#
#================================================================

import os
import keras
import keras.backend as K
import keras_resnet.models
from .. import initializers
from .. import layers
from .. import losses

import numpy as np
import tensorflow as tf
"""
A dictionary mapping custom layer names to the correct classes.
"""
custom_objects = {
    'UpsampleLike': layers.UpsampleLike,
    'PriorProbability': initializers.PriorProbability,
    'RegressBoxes': layers.RegressBoxes,
    'NonMaximumSuppression': layers.NonMaximumSuppression,
    'Anchors': layers.Anchors,
    'ClipBoxes': layers.ClipBoxes,
    '_smooth_l1': losses.smooth_l1(),
    '_focal': losses.focal(),
    'NormalizeBoxes': layers.NormalizeBoxes,
    'TopRoiAligns': layers.TopRoiAligns
}

from .retinanet import retinanet


def __build_local_roi_feature_extraction_model(
        extraction_feature_channels=256):
    """ extract local roi features of FPN to a flattened feature for classification.

    Args:
        roi_features: Tensor of shape [batch_size, max_proposals, pool_shape[0], pool_shape[1], depth]. the top-N roi features.

    Kwargs:
        extraction_feature_channels: Int. the conv layer's output channel

    Returns: Tensor of shape [batch_size, local_feature_length]. The local feature.

    """

    def roi_feature_extraction_model(extraction_feature_channels):
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }
        inputs = keras.layers.Input(shape=(7, 7, extraction_feature_channels))
        outputs = inputs
        # four conv layers
        for i in range(4):
            outputs = keras.layers.Conv2D(
                filters=extraction_feature_channels,
                activation='relu',
                name='roi_feature_extraction_{}'.format(i),
                kernel_initializer=keras.initializers.normal(
                    mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros',
                **options)(outputs)
        outputs = keras.layers.GlobalAvgPool2D(
            name='roi_feature_extraction_pooling')(outputs)
        return keras.models.Model(inputs=inputs, outputs=outputs)

    input1 = keras.layers.Input(shape=(8, 7, 7, extraction_feature_channels))
    extraction_model = roi_feature_extraction_model(
        extraction_feature_channels)
    local_features = keras.layers.TimeDistributed(extraction_model)(
        input1)  # shape: [None, max_proposals, channels]
    local_features = keras.layers.Flatten()(
        local_features)  # shape: [None, local_feature_length]
    return keras.models.Model(inputs=input1, outputs=local_features)


def nspnet(inputs,
           num_classes,
           stage=2,
           stage1_weights_path=None,
           fix_stage1_layers=True,
           global_cls=False,
           **kwargs):
    """TODO: Docstring for nsp_model.

    Args:
        inputs (TODO): TODO
        num_classes (TODO): TODO
        **kwargs (TODO): TODO

    Kwargs:
        name (TODO): TODO
        global_cls (TODO): TODO

    Returns: TODO

    """
    model, features, global_feature = retinanet(
        inputs=inputs, num_classes=num_classes, **kwargs)

    # we expect the anchors, regression and classification values as first output
    anchors = model.outputs[0]
    regression = model.outputs[1]
    classification = model.outputs[2]
    Global_cls = model.outputs[3]
    other = []

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([inputs, boxes])
    normalize_boxes = layers.NormalizeBoxes(name='normalize_boxes')(
        [inputs, boxes])

    # feature_list = features[0:4]  # [P3, P4, P5, P6]
    feature_list = features[0:5]  # [P3, P4, P5, P6, P7]
    nms_classification = layers.NonMaximumSuppression(name='nms', trainable=False)(
        [boxes, classification])

    layer_config = {
        'pool_shape': [7, 7],
        'max_proposals': 8,
        # 'pool_type': 'MAX'
    }
    selected_roi_feature_list = [
        layers.TopRoiAligns(
            name='TopRoiALigns_level{}'.format(i), trainable=False,
            **layer_config)([feature, normalize_boxes, nms_classification])
        for i, feature in enumerate(feature_list)
    ]

    selected_roi_features = layers.SelectiveROI(
        pool_type='MAX')(selected_roi_feature_list)

    local_feature_model = __build_local_roi_feature_extraction_model(
        extraction_feature_channels=256)
    local_feature = local_feature_model(selected_roi_features)
    # local_feature = keras.layers.TimeDistributed(keras.layers.Conv2D(256, (3,3) ) )(selected_roi_features)
    print (global_feature) 
    print(local_feature)
    print (global_feature.get_shape() ) 
    print (local_feature.get_shape() ) 

    # classification_feature = keras.layers.Concatenate()(
        # [global_feature, local_feature])
    classification_feature = keras.layers.Concatenate()(
        [local_feature, global_feature])

    nsp_classification = keras.layers.Dense(
        3, name='fusion_classification')(classification_feature)

    #build stage1 model
    stage1_outputs = [
        regression, classification, boxes, nms_classification, Global_cls
    ]
    stage1_model = keras.models.Model(
        inputs=inputs, outputs=stage1_outputs, name='nsp-stage1')

    if stage1_weights_path != None and os.path.isfile(stage1_weights_path):
        stage1_model.load_weights(stage1_weights_path)
        print(
            'successfully load stage1 weights: {}'.format(stage1_weights_path))

    if fix_stage1_layers:
        for layer in stage1_model.layers:
            layer.trainable = False

    #build stage2 model
    stage2_outputs = [
        # regression, classification, boxes, nms_classification,
        # nsp_classification, Global_cls
        nsp_classification
    ]
    stage2_model = keras.models.Model(inputs=inputs, outputs=stage2_outputs)

    if stage == 1:
        return stage1_model
    else:
        return stage2_model


def resnet50_nspnet(num_classes, inputs=None, **kwargs):
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))
    resnet = keras_resnet.models.ResNet50(
        inputs, include_top=False, freeze_bn=True)
    return nspnet(
        inputs=inputs,
        num_classes=num_classes,
        backbone_layers=resnet.outputs[1:],
        **kwargs)
