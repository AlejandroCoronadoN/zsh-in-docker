import boto3
import geopandas as gpd
import json
import math

import numpy as np
import sys
import torch
import torchvision.transforms as T


incluia_etl = '../../../incluia-etl/' # Points to the folder that contains 'incluiasrc' folder
sys.path.append(incluia_etl)

import incluiasrc.utils.mosaics_functions as etl
from incluiasrc.utils.request_functions import *

from jal.utils.config import *


def mosaic_models_corners(y=None, x=None, img_size=None):
    """
    Given a centroid of an image (y,x), returns a geopandas box of the corresponding generated mosaic.
    Args:
        y: y-centroid coordinate (float)
        x: x-centroid coordinate (float)
        img_size: Number of pixels of the length of the squared image.

    Returns: Geopandas Polygon box of the corresponding generated mosaic.
    """

    box_out = etl.mosaic_models_corners(y=y, x=x, img_size=img_size,
                                        pixel_size_y=pixel_size_y,
                                        pixel_size_x=pixel_size_x,
                                        )
    return box_out
