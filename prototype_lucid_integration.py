#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:14:46 2020

Use lucid to extract filter information by using Feature Visualization techniques

"""

# In[Imports]
import os
import sys


# In[Lucid Imports]
cwd = os.getcwd()
lucid_path = os.path.abspath(os.path.join(cwd, "..", "tools", "lucid"))
    # Using git repository because documentation to load/save models is not available
    # for the current version in pip (0.38)

if not lucid_path in sys.path:
    sys.path.append(lucid_path)

from lucid.modelzoo.vision_base import Model


# In[Setup]
lucid_model_name = "lucid_model.pb"
output_model = "output_model.pb"


# In[Load models]
frozen_model = Model.load_from_metadata(
    output_model,
    metadata = {
        "input_name" : "conv2d", # model.layers[0].name,
        "image_shape" : [10, 400, 1], # input_shape,
        "image_value_range" : [-1, 1]
        }
    )
lucid_model = Model.load(lucid_model_name)

