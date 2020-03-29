#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:55:51 2020

@author: breixo
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt


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


def visualize_timeseries(X_ts, title = None):
    n_ch, n_points = X_ts.shape[0:2]
    
    bg_cmap = matplotlib.cm.get_cmap('seismic')
    bg_colors = bg_cmap(X_ts/2 +0.5)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(20,(1+0.7 *n_ch*2)))
    x_bar = range(n_points)
    for i in range(n_ch):
        ax.bar(
            x = x_bar,
            height = 2,
            width = 1,
            bottom = -2*i-1,
            color = bg_colors[i,:,0],
            alpha = 0.5,
            edgecolor = None)
        ax.plot((X_ts[i,:] - i*2), color="black")
        
    if title:
        ax.set_title(title)
    # plt.yticks(-np.arange(n_ch)*2, ['channel ' + str(i) for i in range(n_ch)])
    ax.set_yticks(-np.arange(n_ch)*2)
    ax.set_yticklabels(['channel ' + str(i) for i in range(n_ch)])
    ax.set_xlabel('time')
    return ax
