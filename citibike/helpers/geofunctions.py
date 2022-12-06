"""Functions related to geo locations etc."""
__author__ = 'Felix Kleinert'
__date__ = '2021-02-16'

import dask.array as da
import numpy as np
import xarray as xr

from citibike.helpers import convert2xrda
from typing import Union, Tuple

xr_int_float = Union[xr.DataArray, xr.Dataset, np.ndarray, int, float]
tuple_of_2xr_int_float = Tuple[xr_int_float, xr_int_float]


def deg2rad_all_points(lat1: xr_int_float, lon1: xr_int_float,
                       lat2: xr_int_float, lon2: xr_int_float) -> Tuple[tuple_of_2xr_int_float, tuple_of_2xr_int_float]:
    """
    Converts coordinates provided in lat1, lon1, lat2, and lon2 from deg to rad. In fact this method just calls
    dasks deg2rad method on all inputs and returns a tuple of tuples.

    :param lat1: Latitude(-s) of first location
    :type lat1:
    :param lon1: Longitude(-s) of first location
    :type lon1:
    :param lat2: Latitude(-s) of second location
    :type lat2:
    :param lon2: Longitude(-s) of second location
    :type lon2:
    :return: Lats and lons in radians ((lat1, lon1), (lat2, lon2))
    :rtype:
    """
    lat1, lon1, lat2, lon2 = da.deg2rad(lat1), da.deg2rad(lon1), da.deg2rad(lat2), da.deg2rad(lon2)
    return (lat1, lon1), (lat2, lon2)


def haversine_dist(lat1, lon1,
                   lat2, lon2,
                   to_radians: bool = True, earth_radius: float = 6371.,):
    """
    Calculate the great circle distance between two points
    on the Earth (specified in decimal degrees or in radians)

    Reference: ToBeAdded
    (First implementation provided by M. Langguth)

    :param lat1: Latitude(-s) of first location
    :param lon1: Longitude(-s) of first location
    :param lat2: Latitude(-s) of second location
    :param lon2: Longitude(-s) of second location
    :param to_radians: Flag if conversion from degree to radiant is required
    :param earth_radius: Earth radius in kilometers
    :return: Distance between locations in kilometers
    """

    if to_radians:
        (lat1, lon1), (lat2, lon2) = deg2rad_all_points(lat1, lon1, lat2, lon2)

    #lat1 = convert2xrda(lat1, use_1d_default=True)
    #lon1 = convert2xrda(lon1, use_1d_default=True)
    #lat2 = convert2xrda(lat2, use_1d_default=True)
    #lon2 = convert2xrda(lon2, use_1d_default=True)

    #assert lat1.shape == lon1.shape
    #assert lat2.shape == lon2.shape
    #assert isinstance(lat1, xr.DataArray)
    #assert isinstance(lon1, xr.DataArray)
    #assert isinstance(lat2, xr.DataArray)
    #assert isinstance(lon2, xr.DataArray)
    #assert len(lat1.shape) >= len(lat2.shape)

    # broadcast lats and lons to calculate distances in a vectorized manner.
    #lat1, lat2 = xr.broadcast(lat1, lat2)
    #lon1, lon2 = xr.broadcast(lon1, lon2)

    a = da.sin((lat2 - lat1) / 2.0) ** 2 + \
        da.cos(lat1) * da.cos(lat2) * da.sin((lon2 - lon1) / 2.0) ** 2

    dist = earth_radius * 2. * da.arcsin(da.sqrt(a))

    return dist

