import logging
import os
import copy

from matplotlib import pyplot as plt
from citibike.helpers import TimeTracking


class AbstractPlotClass:  # pragma: no cover
    """
    Abstract class for all plotting routines to unify plot workflow.

    Taken from MLAir:
    https://gitlab.jsc.fz-juelich.de/esde/machine-learning/mlair/-/blob/master/mlair/plotting/abstract_plot_class.py)

    Each inheritance requires a _plot method. Create a plot class like:

    .. code-block:: python

        class MyCustomPlot(AbstractPlotClass):

            def __init__(self, plot_folder, *args, **kwargs):
                super().__init__(plot_folder, "custom_plot_name")
                self._data = self._prepare_data(*args, **kwargs)
                self._plot(*args, **kwargs)
                self._save()

            def _prepare_data(*args, **kwargs):
                <your custom data preparation>
                return data

            def _plot(*args, **kwargs):
                <your custom plotting without saving>

    The save method is already implemented in the AbstractPlotClass. If special saving is required (e.g. if you are
    using pdfpages), you need to overwrite it. Plots are saved as .pdf with a resolution of 500dpi per default (can be
    set in super class initialisation).

    Methods like the shown _prepare_data() are optional. The only method required to implement is _plot.

    If you want to add a time tracking module, just add the TimeTrackingWrapper as decorator around your custom plot
    class. It will log the spent time if you call your plotting without saving the returned object.

    .. code-block:: python

        class MyCustomPlot(AbstractPlotClass):
            pass

    """

    def __init__(self, plot_folder, plot_name, resolution=500, rc_params=None):
        """Set up plot folder and name, and plot resolution (default 500dpi)."""
        plot_folder = os.path.abspath(plot_folder)
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        self.plot_folder = plot_folder
        self.plot_name = plot_name.replace("/", "_") if plot_name is not None else plot_name
        self.plot_name_base = copy.deepcopy(self.plot_name)
        self.resolution = resolution
        if rc_params is None:
            rc_params = {'axes.labelsize': 'small',
                         'xtick.labelsize': 'x-small',
                         'ytick.labelsize': 'x-small',
                         'legend.fontsize': 'small',
                         'axes.titlesize': 'small',
                         }
        self.rc_params = rc_params
        self._update_rc_params()

    def __del__(self):
        try:
            plt.close('all')
        except ImportError:
            pass

    def plot(self, *args, **kwargs):
        with TimeTracking(name=self.plot.__qualname__):
            self._plot(*args, **kwargs)
            self._save(bbox_inches='tight')

    def _plot(self, *args, **kwargs):
        """Abstract plot class needs to be implemented in inheritance."""
        raise NotImplementedError

    def _save(self, **kwargs):
        """Store plot locally. Name of and path to plot need to be set on initialisation."""
        plot_name = os.path.join(self.plot_folder, f"{self.plot_name}.pdf")
        logging.debug(f"... save plot to {plot_name}")
        plt.savefig(plot_name, dpi=self.resolution, **kwargs)
        plt.close('all')

    def _update_rc_params(self):
        plt.rcParams.update(self.rc_params)

    @staticmethod
    def _get_sampling(sampling):
        if sampling == "daily":
            return "D"
        elif sampling == "hourly":
            return "h"

    @staticmethod
    def get_dataset_colors():
        """
        Standard colors used for train-, val-, and test-sets during postprocessing
        """
        colors = {"train": "#e69f00", "val": "#009e73", "test": "#56b4e9", "train_val": "#000000"}  # hex code
        return colors

