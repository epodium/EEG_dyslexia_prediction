#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:55:51 2020

@author: breixo
"""

import numpy as np
from matplotlib import pyplot as plt, cm


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
        ax = fig.add_subplot(2, 1, 1)
        visualize_timeseries(network_input, ax = ax)
        # fig.add_subplot(1, 2, 1)
        plt.axis('off')
        # plt.imshow(np.transpose(network_input))
        # plt.imshow(network_input.reshape(network_input.shape[0:2]))
        # plt.imshow(np.repeat(network_input.reshape(network_input.shape[0:2]), 8, 0))
        # plt.colorbar(ticks=ticks, orientation='horizontal')
        
    gradcam_floored = np.maximum(gradcam, 0)

    # fig.add_subplot(n_halves + 1, n_plots, n_halves*n_plots +1)
    ax = fig.add_subplot(n_plots, n_halves + 1, n_halves*n_plots +1)
    plt.axis('off')
    # plt.imshow(np.repeat(gradcam_floored, 16, 0))
    superpose_gradcam(gradcam_floored, network_input, ax = ax)
    # plt.imshow(gradcam_floored)
    # plt.colorbar(ticks=ticks, orientation='horizontal')
    
    # fig.add_subplot(n_halves +1, n_plots, n_halves*n_plots +2)
    ax = fig.add_subplot(n_plots, n_halves +1, n_halves*n_plots +2)
    plt.axis('off')
    # plt.imshow(np.repeat(gradcam, 16, 0))
    superpose_gradcam(gradcam, network_input, ax = ax)
    # plt.imshow(gradcam)
    # plt.colorbar(ticks=ticks, orientation='horizontal')
    # fig.show()


def visualize_timeseries(X_ts, title = None, ax = None):
    n_ch, n_points = X_ts.shape[0:2]
    
    bg_cmap = cm.get_cmap('seismic')
    bg_colors = bg_cmap(X_ts/2 +0.5)

    if ax is None:
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
            alpha = 0.75,
            edgecolor = None)
        ax.plot((X_ts[i,:] - i*2), color="black")
        
    if title:
        ax.set_title(title)
    ax.set_yticks(-np.arange(n_ch)*2)
    ax.set_yticklabels(['channel ' + str(i) for i in range(n_ch)])
    ax.set_xlabel('time')
    return ax

def superpose_gradcam(
        gradcam,
        network_input,
        ax = None):
    n_ch, w_input = network_input.shape[0:2]
    
    bg_cmap = cm.get_cmap('inferno')
    
    norm = cm.colors.Normalize(vmin = np.min(gradcam), vmax = np.max(gradcam))
    bg_colors = bg_cmap(norm(gradcam))
    
    if ax is None:
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(20,(1+0.7 *n_ch*2)))
    
    print(gradcam.shape)
    w_gradcam = gradcam.shape[1]
    step = (w_input-1)/(w_gradcam-1) #TODO Is this the correct way to approach this?
    gradcam_bar = np.arange(w_gradcam) * step
    for i in range(n_ch):
        ax.bar(
            x = gradcam_bar,
            height = 2,
            width = step,
            bottom = -2*i-1,
            color = bg_colors[i,:],
            alpha = 0.75,
            edgecolor = None)
        ax.plot((network_input[i, :] - i*2), color="black")
        
    ax.set_yticks(-np.arange(n_ch)*2)
    ax.set_yticklabels(['channel ' + str(i) for i in range(n_ch)])
    ax.set_xlabel('time')
    
    ax.figure.colorbar(
        cm.ScalarMappable(norm=norm, cmap=bg_cmap),
        ax=ax,
        orientation='horizontal')