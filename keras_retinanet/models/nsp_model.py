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

import keras
from .. import initializers
from .. import layers
from .. import losses

import numpy as np
"""
A dictionary mapping custom layer names to the correct classes.
"""
custom_objects = {
    'UpsampleLike':          layers.UpsampleLike,
    'PriorProbability':      initializers.PriorProbability,
    'RegressBoxes':          layers.RegressBoxes,
    'NonMaximumSuppression': layers.NonMaximumSuppression,
    'Anchors':               layers.Anchors,
    'ClipBoxes':             layers.ClipBoxes,
    '_smooth_l1':            losses.smooth_l1(),
    '_focal':                losses.focal(),
    'NormalizeBoxes':        layers.NormalizeBoxes,
    'SelectiveROIAlign':     layers.SelectiveROIAlign
}

from .retinanet import retinanet


def nsp_model(inputs,
              num_classes,
              name='nsp-model',
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
    model = retinanet(inputs=inputs, num_classes=num_classes, **kwargs)

    # we expect the anchors, regression and classification values as first output
    anchors        = model.outputs[0]
    regression     = model.outputs[1]
    classification = model.outputs[2]
    Global_cls = model.outputs[3]
    other = [] 

    # apply predicted regression to anchors
    boxes           = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes           = layers.ClipBoxes(name='clipped_boxes')([inputs, boxes])
    normalize_boxes = layers.NormalizeBoxes(name= 'normalize_boxes')([inputs, boxes] )

    nms_classification  = layers.NonMaximumSuppression(name='nms')([boxes, classification])
    selected_rois = layers.SelectiveROIAlign(pool_shape= [7,7], max_proposals=10, name= 'selective roi align') 
