#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:55:51 2020

@author: breixo
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# In[Define Visualize Grad_Cam]

def visualize_gradcam(
        gradcam,
        network_input = None,
        label = None,
        layer = None):
    fig = plt.figure(figsize = (16, 6.4))
    plt.axis('off')
    ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    n_plots = 2
    n_halves = 0
    title = ""
    if label is not None:
        title += f", class {label}"
    if layer is not None:
        title += f", layer {layer}"
    plt.title(f'GradCAM{title}')
    if network_input is not None:
        n_halves = 1
        fig.add_subplot(2, 1, 1)
        # fig.add_subplot(1, 2, 1)
        plt.axis('off')
        # plt.imshow(np.transpose(network_input))
        # plt.imshow(network_input.reshape(network_input.shape[0:2]))
        plt.imshow(np.repeat(network_input.reshape(network_input.shape[0:2]), 8, 0))
        plt.colorbar(ticks=ticks, orientation='horizontal')
        
    gradcam_floored = np.maximum(gradcam, 0)

    # fig.add_subplot(n_halves + 1, n_plots, n_halves*n_plots +1)
    fig.add_subplot(n_plots, n_halves + 1, n_halves*n_plots +1)
    plt.axis('off')
    plt.imshow(np.repeat(gradcam_floored, 16, 0))
    # plt.imshow(gradcam_floored)
    plt.colorbar(ticks=ticks, orientation='horizontal')
    
    # fig.add_subplot(n_halves +1, n_plots, n_halves*n_plots +2)
    fig.add_subplot(n_plots, n_halves +1, n_halves*n_plots +2)
    plt.axis('off')
    plt.imshow(np.repeat(gradcam, 16, 0))
    # plt.imshow(gradcam)
    plt.colorbar(ticks=ticks, orientation='horizontal')
    # fig.show()

