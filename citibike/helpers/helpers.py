"""Track time either as decorator or explicit. Taken from MLAir"""
import datetime as dt
import logging
import math
import time
import types
from functools import wraps
from typing import Optional
import numpy as np
import xarray as xr
import dask.array as da
import inspect


from typing import Dict, Callable, Union, List, Any, Tuple

"""
This helper file is copy-pasted from MLAir (https://gitlab.jsc.fz-juelich.de/esde/machine-learning/mlair)

"""



class TimeTrackingWrapper:
    r"""
    Wrapper implementation of TimeTracking class.

    Use this implementation easily as decorator for functions, classes and class methods. Implement a custom function
    and decorate it for automatic time measure.

    .. code-block:: python

        @TimeTrackingWrapper
        def sleeper():
            print("start")
            time.sleep(1)
            print("end")

        >>> sleeper()
        start
        end
        INFO: foo finished after 00:00:01 (hh:mm:ss)

    """

    def __init__(self, func):
        """Construct."""
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        """Start time tracking."""
        with TimeTracking(name=self.__wrapped__.__name__):
            return self.__wrapped__(*args, **kwargs)

    def __get__(self, instance, cls):
        """Create bound method object and supply self argument to the decorated method."""
        if instance is None:
            return self
        else:
            return types.MethodType(self, instance)


class TimeTracking(object):
    """
    Track time to measure execution time.

    Time tracking automatically starts on initialisation and ends by calling stop method. Duration can always be shown
    by printing the time tracking object or calling get_current_duration. It is possible to start and stop time tracking
    by hand like

    .. code-block:: python

        time = TimeTracking(start=True)  # start=True is default and not required to set
        do_something()
        time.stop(get_duration=True)

    A more comfortable way is to use TimeTracking in a with statement like:

    .. code-block:: python

        with TimeTracking():
            do_something()

    The only disadvantage of the latter implementation is, that the duration is logged but not returned.
    """

    def __init__(self, start=True, name="undefined job", logging_level=logging.INFO, log_on_enter=False):
        """Construct time tracking and start if enabled."""
        self.start = None
        self.end = None
        self._name = name
        self._logging = {logging.INFO: logging.info, logging.DEBUG: logging.debug}.get(logging_level, logging.info)
        self._log_on_enter = log_on_enter
        if start:
            self._start()

    def _start(self) -> None:
        """Start time tracking."""
        self.start = time.time()
        self.end = None

    def _end(self) -> None:
        """Stop time tracking."""
        self.end = time.time()

    def _duration(self) -> float:
        """Get duration in seconds."""
        if self.end:
            return self.end - self.start
        else:
            return time.time() - self.start

    def __repr__(self) -> str:
        """Display current passed time."""
        return f"{dt.timedelta(seconds=math.ceil(self._duration()))} (hh:mm:ss)"

    def run(self) -> None:
        """Start time tracking."""
        self._start()

    def stop(self, get_duration=False) -> Optional[float]:
        """
        Stop time tracking.

        Will raise an error if time tracking was already stopped.
        :param get_duration: return passed time if enabled.

        :return: duration if enabled or None
        """
        if self.end is None:
            self._end()
        else:
            msg = f"Time was already stopped {time.time() - self.end}s ago."
            raise AssertionError(msg)
        if get_duration:
            return self.duration()

    def duration(self) -> float:
        """Return duration in seconds."""
        return self._duration()

    def __enter__(self):
        """Context manager."""
        self._logging(f"start {self._name}") if self._log_on_enter is True else None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop time tracking on exit and log info about passed time."""
        self.stop()
        self._logging(f"{self._name} finished after {self}")


def replace_space(s: str, by='_'):
    return s.replace(' ', by)


def to_list(obj: Any) -> List:
    """
    Transform given object to list if obj is not already a list. Sets are also transformed to a list.

    :param obj: object to transform to list

    :return: list containing obj, or obj itself (if obj was already a list)
    """
    if isinstance(obj, (set, tuple, type({}.keys()))):
        obj = list(obj)
    elif not isinstance(obj, list):
        obj = [obj]
    return obj


def is_xarray(arr) -> bool:
    """
    Returns True if arr is xarray.DataArray or xarray.Dataset.
    :param arr: variable in question
    :type arr: Any
    :return:
    :rtype: bool
    """
    return isinstance(arr, xr.DataArray) or isinstance(arr, xr.Dataset)


def convert2xrda(arr: Union[xr.DataArray, xr.Dataset, np.ndarray, int, float],
                 use_1d_default: bool = False, **kwargs) -> Union[xr.DataArray, xr.Dataset]:
    """
    Converts np.array, int or float object to xarray.DataArray.

    If a xarray.DataArray or xarray.Dataset is passed, returns that unchanged.
    :param arr:
    :type arr: xr.DataArray, xr.Dataset, np.ndarray, int, float
    :param use_1d_default:
    :type use_1d_default: bool
    :param kwargs: Any additional kwargs which are accepted by xr.DataArray()
    :type kwargs:
    :return:
    :rtype: xr.DataArray, xr.DataSet
    """
    if is_xarray(arr):
        return arr
    else:
        if use_1d_default:
            if isinstance(arr, da.core.Array):
                raise TypeError(f"`use_1d_default=True' is used with `arr' of type da.array. For da.arrays please "
                                f"pass `use_1d_default=False' and specify keywords for xr.DataArray via kwargs.")
            dims = kwargs.pop('dims', 'points')
            coords = kwargs.pop('coords', None)
            try:
                if coords is None:
                    coords = {dims: range(arr.shape[0])}
            except (AttributeError, IndexError):
                if isinstance(arr, int) or isinstance(arr, float):
                    coords = kwargs.pop('coords', {dims: range(1)})
                    dims = to_list(dims)
                else:
                    raise TypeError(f"`arr' must be arry-like, int or float. But is of type {type(arr)}")
            kwargs.update({'dims': dims, 'coords': coords})

        return xr.DataArray(arr, **kwargs)


def to_list(obj: Any) -> List:
    """
    Transform given object to list if obj is not already a list. Sets are also transformed to a list.

    :param obj: object to transform to list

    :return: list containing obj, or obj itself (if obj was already a list)
    """
    if isinstance(obj, (set, tuple, type({}.keys()))):
        obj = list(obj)
    elif not isinstance(obj, list):
        obj = [obj]
    return obj


def remove_items(obj: Union[List, Dict, Tuple], items: Any):
    """
    Remove item(s) from either list, tuple or dictionary.

    :param obj: object to remove items from (either dictionary or list)
    :param items: elements to remove from obj. Can either be a list or single entry / key

    :return: object without items
    """

    def remove_from_list(list_obj, item_list):
        """Remove implementation for lists."""
        if len(item_list) > 1:
            return [e for e in list_obj if e not in item_list]
        elif len(item_list) == 0:
            return list_obj
        else:
            list_obj = list_obj.copy()
            try:
                list_obj.remove(item_list[0])
            except ValueError:
                pass
            return list_obj

    def remove_from_dict(dict_obj, key_list):
        """Remove implementation for dictionaries."""
        return {k: v for k, v in dict_obj.items() if k not in key_list}

    items = to_list(items)
    if isinstance(obj, list):
        return remove_from_list(obj, items)
    elif isinstance(obj, dict):
        return remove_from_dict(obj, items)
    elif isinstance(obj, tuple):
        return tuple(remove_from_list(to_list(obj), items))
    else:
        raise TypeError(f"{inspect.stack()[0][3]} does not support type {type(obj)}.")


def standardize(df, mean=None, std=None):
    if mean is None:
        mean = df.mean(axis=0)
    if std is None:
        std = df.std(axis=0)

    res = (df - mean)/std
    return res, mean, std
