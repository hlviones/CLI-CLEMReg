#!/usr/bin/env python3
# coding: utf-8
import time
from skimage import feature
import numpy as np

def point_cloud_sampling(input,
                         sampling_frequency: float = 0.01,
                         sigma: float = 1.0,
                         edge_color: str = 'black',
                         face_color: str = 'red',
                         point_size: int = 5):
    """
    ?

    Parameters
    ----------
    edge_color : str
        ?
    input : napari.layers.Labels
        ?
    sampling_frequency : float
        Frequency of cloud sampling
    sigma : float
        ?
    face_color : str
        ?
    point_size : int
        ?
    Returns
    -------
    #
    """
    print(f'Sampling point cloud from {input.name} with sigma={sigma} and sampling_frequency={sampling_frequency}...')
    start_time = time.time()

    point_lst = []

    data = np.array(input.data)
    print(f"data shape: {data.shape}") 
    for z in range(data.shape[0]):
        img = (data[z] > 0).astype('uint8') * 255
        img = feature.canny(img, sigma=sigma)
        img = feature.canny(img, sigma=sigma)
        points = np.where(img == 1)
        # Just keep values at every n-th position
        for i in range(len(points[0])):
            if np.random.rand() < sampling_frequency:
                point_lst.append([z, points[0][i], points[1][i]])

    kwargs = dict(
        edge_color=edge_color,
        face_color=face_color,
        size=point_size,
        name=input.name + '_points'
    )

    print(f'Finished point cloud sampling after {time.time() - start_time}s!')

    return np.asarray(point_lst), kwargs
