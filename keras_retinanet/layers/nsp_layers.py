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

import numpy as np

class NormalizedBoxes(keras.layers.Layer):

    """normalize the values of boxes to [0,1]"""

    def __init__(self, **kwargs):
        """TODO: to be defined1. """

        

class SelectiveROIAlign(keras.layers.Layer):

    """Implements Selective ROIAlign. merge multi-level ROI regoins of feature maps"""

    def __init__(self, pool_shape, **kwargs):
        """ class initializers

        Args:
            pool_shape: a list of rank 2, representing [height, width] of the output pooled regions. Usually [7,7] 
            **kwargs: other optional arguments defined in @keras.layers.Layer
        """
        super(SelectiveROIAlign, self).__init__(**kwargs)  

        self._pool_shape = pool_shape
        
    def call(self, inputs, **kwargs):
        """

        Args:
            inputs: a list of tensors: [features, boxes, nms_classification, image]. features is 
            **kwargs: other args

        Returns: TODO

        """
        pass    
        
