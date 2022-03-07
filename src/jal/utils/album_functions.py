import boto3
import geopandas as gpd
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image

incluia_etl = '../../../incluia-etl/' # Points to the folder that contains 'incluiasrc' folder
sys.path.append(incluia_etl)

from jal.utils.config import *

from incluiasrc.utils.request_functions import *
import incluiasrc.utils.album_functions as etl

# Load album from local or AWS S3
local_album_dir = '../data/02_raw_images_metadata/'
s3_album_dir = local_album_dir.replace('../data/', '')

album_name = 'album.geojson'
local_album_path = local_album_dir + album_name
s3_album_path = s3_album_dir + album_name

s3_client = boto3.client('s3')

try:
    album_dict = json.load(open(local_album_path))
    album = gpd.GeoDataFrame.from_features(album_dict["features"], crs='EPSG:4326')
except:
    album = get_gpd(bucket, s3_album_path)

# number of cells in each batch
x_batch_dim = album['cell_x'].max() + 1
y_batch_dim = album['cell_y'].max() + 1

# number of batches in shapefile
x_shp_dim = album['batch_x'].max() + 1
y_shp_dim = album['batch_y'].max() + 1

# To obtain these numbers run find_pixel_intersection() function and save output values here.
# number of horizontal pixels intersection across images
pixels_intersect_h = -9  # find_pixel_intersection(horizontal=True)  # Run in jn, not here! Range=[-1,-375]
# number of vertical pixels intersection across images
pixels_intersect_v = 11  # find_pixel_intersection(horizontal=False) #  Range=[0, 399]

album_batch = album[album['batch_label'] == '142_120']
album_batch_lat = album_batch.centroid_lat
album_batch_lon = album_batch.centroid_lon
batch_corner_ctr = np.zeros(4)
batch_corner_ctr[0] = float(album_batch_lon.min())
batch_corner_ctr[1] = float(album_batch_lat.max())
batch_corner_ctr[2] = float(album_batch_lon.max())
batch_corner_ctr[3] = float(album_batch_lat.min())

# pixels between two contiguous centroids (ctr)
pixels_between_ctr_y = 366  # 375 + pixels_intersect_h
pixels_between_ctr_x = 388  # 400 - pixels_intersect_v - 1
# To understand why we subtract -1, let's suppose that pixels_intersect_h = -1, and pixels_intersect_x = 0,
# then it makes perfect sense that pixels_between_ctr_y = 374 and pixels_between_ctr_x = 399.

# pixels between the extreme centroids of a batch
pixels_between_ext_ctr_y = (y_batch_dim - 1) * pixels_between_ctr_y
pixels_between_ext_ctr_x = (x_batch_dim - 1) * pixels_between_ctr_x

# pixel size in the y-direction and x-direction in map units/pixel
pixel_size_y = (batch_corner_ctr[1] - batch_corner_ctr[3]) / pixels_between_ext_ctr_y
pixel_size_x = (batch_corner_ctr[2] - batch_corner_ctr[0]) / pixels_between_ext_ctr_x

# latitude and longitude difference between two contiguous images
lat_difference = pixel_size_y * pixels_between_ctr_y
lon_difference = pixel_size_x * pixels_between_ctr_x


def mosaic_around_point(x, y, size, batch_dict, verbose=True):
    """
    Generates a mosaic around a point using batches. A batch is a large mosaic previously stored in S3.
    ----------
    Parameters
        x : float. Longitude of the point of interest.
        y : float. Latitude of the point of interest.
        size: int. Number of pixels of square's side output.
        batch_dict: dictionary of matrices. Initially, in the jupypter-notebook, batch_dict is initailized as
        batch_dict={}. Then on each mosaic_around_point iteration, the dictionary, which is sent back as an ouput,
        adds if necessary the batches that were necessary for producing the mosaic from that iteration. Since the whole
        city imagery is to memory expensive, the dictionary stores only a number of batches that the machine can store
        in RAM. Therefore, to generate mosaics fast, it is highly recommended that when using it iteratively for a list
        of coordinates, the coordinates are ordered.
        verbose: boolean. Prints verbose.
    """
    
    img, wld, batch_dict = etl.mosaic_around_point(x, y, size, batch_dict,
                                              album=album,
                                              bucket=bucket,
                                              lat_difference=lat_difference,
                                              lon_difference=lon_difference,
                                              pixel_size_y=pixel_size_y,
                                              pixel_size_x=pixel_size_x,
                                              y_batch_dim=y_batch_dim,
                                              pixels_intersect_h=pixels_intersect_h,
                                              pixels_intersect_v=pixels_intersect_v,
                                              verbose=verbose,
                                              )

    return img, wld, batch_dict


def mosaic_around_image(x, y, size):
    """
    Displays a mosaic given a point in coordinates x (longitude) and y (latitude).
    size is used to calculate the number of images the mosaic contains.
    size=1 is a single image, size=2 is a 3x3 mosaic, size=3 is a 5x5 mosaic, etc.
    """
    img = etl.mosaic_around_image(x, y, size,
                                  album=album,
                                  bucket=bucket,
                                  x_batch_dim=x_batch_dim,
                                  y_batch_dim=y_batch_dim,
                                  pixels_intersect_h=pixels_intersect_h,
                                  pixels_intersect_v=pixels_intersect_v,
                                  )
    return img


def find_pixel_intersection(horizontal):
    """
    Find the number of horizontal or vertical pixels that share intersection across images.
    ----------
    Parameters
        horizontal: Boolean. If True finds horizontal intersection strip. If False, finds vertical intersection strip.
        album: geopandas dataframe. Country album.
        bucket: str. Bucket name in AWS
    """
    pixel_min = etl.find_pixel_intersection(horizontal,
                                            album=album,
                                            bucket=bucket,
                                            x_batch_dim=x_batch_dim,
                                            y_batch_dim=y_batch_dim)
    return pixel_min


def batch_generator(batch_label, generate_batch=True, verbose=False):
    """
    Generate a large mosaic of images that correspond to a batch in the album.
    ----------
    Parameters
        batch_label: str. The batch label for which the mosaic is generated.
        generate_batch: Boolean. If False, generates only wld file
        verbose: Warning for images mismatches with nieghboring images.
    """
    img, wld = etl.batch_generator(batch_label,
                                   generate_batch=generate_batch,
                                   verbose=verbose,
                                   album=album,
                                   bucket=bucket,
                                   pixel_size_y=pixel_size_y,
                                   pixel_size_x=pixel_size_x,
                                   x_batch_dim=x_batch_dim,
                                   y_batch_dim=y_batch_dim,
                                   pixels_intersect_h=pixels_intersect_h,
                                   pixels_intersect_v=pixels_intersect_v,
                                   lat_difference=lat_difference,
                                   lon_difference=lon_difference,
                                   )
    return img, wld

