import copy
import logging

import dask
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import verif
from dask import dataframe as dd
from matplotlib import pyplot as plt
from sklearn import metrics

from citibike.helpers import haversine_dist, remove_items, to_list, TimeTrackingWrapper, replace_space
from citibike.plotting.abstract_plot_class import AbstractPlotClass
from run import DateTimeSplitter, batch_size


class PlotDurationDist(AbstractPlotClass):
    '''
    Overview plot (pairplot) containing (at least) trip duration and trip distance
    '''

    def __init__(self, data, additional_vars=None, plot_folder: str = '.', plot_name='duration_dist'):
        super().__init__(plot_folder, plot_name)
        self._fixed_sel = ('tripduration', 'dist')
        if additional_vars is None:
            additional_vars = []
        self.additional_vars = additional_vars
        self.data = self._process_data(data)
        self._ax = None
        self.additional_ntv = None

    def _plot(self, n_samples=100000, **kwargs):
        kind = kwargs.pop('kind', 'hist')

        data_sample = self.data.sample(n_samples, axis=0).reset_index()
        self._ax = sns.pairplot(data_sample[self.selected_vars], kind=kind, **kwargs)
        self._ax.fig.subplots_adjust(top=0.9)
        self._ax.fig.suptitle(f'Sample Overview (n={n_samples})')

    def _process_data(self, data):

        data_dur_dist = data[['tripduration']]
        data_dur_dist['dist'] = haversine_dist(lat1=data['start station latitude'].values,
                                               lon1=data['start station longitude'].values,
                                               lat2=data['end station latitude'].values,
                                               lon2=data['end station longitude'].values
                                               )
        data_dur_dist = data_dur_dist[data_dur_dist <= data_dur_dist.quantile(.99, axis=0)]
        time_splits, used_time_vars = DateTimeSplitter.to_attributes(data, time_split_attributes=self.additional_vars,
                                                              return_used_attributes=True)
        self.additional_ntv = remove_items(self.additional_vars, used_time_vars)  # No Time Vars
        data_dur_dist = dd.concat([data_dur_dist, time_splits, data[self.additional_ntv]], axis=1).dropna()
        data_dur_dist = data_dur_dist[self.selected_vars].compute()
        return data_dur_dist

    @property
    def selected_vars(self):
        sel_vars = to_list(self._fixed_sel) + to_list(self.additional_vars)
        return sel_vars


class PlotFromAtoBHeatmap(AbstractPlotClass):
    '''
    Heatmap showing number of rides from station A to B
    '''

    def __init__(self, data, y_name, x_name, plot_folder: str = '.', plot_name='from_A_to_B_matrix'):
        super().__init__(plot_folder, plot_name)
        self.data = data
        self.y_name = y_name
        self.x_name = x_name
        self._ax = None
        self._gl = None
        self.ddf_processed = self._process_data(data=data, y_name=y_name, x_name=x_name)
        logging.info(f'PlotFromAtoBHeatmap initialized...')

    def _plot(self, log_scale=False, **kwargs):
        label = kwargs.pop('cbar_kws_label', '# of rides')
        robust = kwargs.pop('robust', 'False')
        vmax = None if robust else 4000
        epsilon_log = kwargs.pop('epsilon_log', 0.01)
        self._ax = plt.axes()
        ddf_grp = copy.deepcopy(self.ddf_processed)
        if log_scale:
            ddf_grp = np.log(ddf_grp + epsilon_log)
            label = f'log({label})'
            self.plot_name = f'{self.plot_name_base}_log'
        else:
            self.plot_name = self.plot_name_base
        sns.heatmap(ddf_grp, ax=self._ax, cmap='BuPu', vmin=0, vmax=vmax, robust=robust,
                    cbar_kws={'label': label, 'extend': 'max'})
        self._ax.set_title('From A to B')


    @staticmethod
    def _process_data(data, y_name, x_name):
        data_selected = data[[y_name, x_name]]
        ddf_grp = data_selected.groupby([y_name, x_name]).size().reset_index().compute().pivot(
            index=y_name, columns=x_name)
        ddf_grp.columns = ddf_grp.columns.droplevel(0)
        return ddf_grp


class PlotNumRidesPerXY(AbstractPlotClass):
    '''
    Number of rides per selected feature
    '''

    def __init__(self, data, plot_folder: str = '.', plot_name='number_of_rides_per'):
        super().__init__(plot_folder, plot_name)
        self.data = data
        self.__allowed_categories = {'usertype': ['Subscriber', 'Customer'],
                                     'gender': ['0', '1', '2']}
        self._ax = None

    @TimeTrackingWrapper
    def _plot(self, per_col_name, category=None, min_idx_value=None, normalize=False):

        title = self.get_plot_title(min_idx_value, per_col_name, False)
        df = self._process_data(per_col_name, category, min_idx_value, normalize)

        self._ax = plt.axes()
        df.plot(kind='bar', stacked=True, ax=self._ax, title=title)
        if category is not None:
            self._ax.get_legend().set_title(category)
        self._ax.set_xlabel(per_col_name)
        cat_post = '' if category is None else f'_{category}'
        norm_post = '_normalized' if normalize else ''
        self.plot_name = f'{self.plot_name_base}_{replace_space(per_col_name)}{cat_post}{norm_post}'

    @staticmethod
    def get_plot_title(min_idx_value, per_col_name, normalize):
        norm_post = '(normalized)' if normalize else ''
        if min_idx_value is None:
            title = f'# rides per {per_col_name} {norm_post}'
        else:
            title = fr'# rides per {per_col_name} ($\geq$ {min_idx_value} {norm_post})'
        return title

    def _process_data(self, per_col_name, category, min_idx_value=None, normalize=False):
        if category is None:
            counts = self.data[per_col_name].value_counts(normalize=normalize)
            counts_c = dask.compute(counts)[0]
        else:
            counts = (self.data[per_col_name].loc[self.data[category] == v].value_counts(normalize=normalize).rename(v)
                      for v in self.__allowed_categories[category])
            counts_c = dask.compute(*counts)
            counts_c = pd.concat(counts_c, axis=1)
        counts_c = counts_c.sort_index().fillna(0).astype(np.int64)
        if min_idx_value is not None:
            counts_c = counts_c[counts_c.index >= min_idx_value]
        return counts_c


class PlotROC(AbstractPlotClass):

    def __init__(self, plot_folder: str = '.', plot_name='roc'):
        super().__init__(plot_folder, plot_name)
        self._ax = None

    def _plot(self, model, X_data, y_data):
        self._ax = metrics.plot_roc_curve(model, X_data, y_data)


class PlotROCs(AbstractPlotClass):
    '''
    Receiver Operating Characteristic curve for multiple classification methods.

    '''

    def __init__(self, batch_size=10000, plot_folder: str = '.', plot_name='roc_comparisons'):
        super().__init__(plot_folder, plot_name)
        self.batch_size = batch_size
        self._ax = None
        self._fig = None

    def _plot(self, y_preds, y_obs):
        self._fig = plt.figure(1)
        for name, y_pred in y_preds.items():
            logging.info(f'Running ROC and AUC for model: {name}')
            try:
                fpr, tpr, thresholds = metrics.roc_curve(y_obs, y_pred)
            except ValueError:
                fpr, tpr, thresholds = metrics.roc_curve(y_obs, y_pred[:, 1])
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (area = {auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.legend(loc='best')
        plt.xlabel("False Positive Rate (Positive label: 1")
        plt.ylabel("True Positive Rate (Positive label: 1)")
        plt.title('ROC curve comparison')

    def _plot2(self, models, X_data, y_data, names=None):
        self._fig = plt.figure(1)
        for name, model in zip(to_list(names), models):
            logging.info(f'Running ROC and AUC for model: {name}')
            # duck typing for keras/sklearn models
            try:
                y_pred = model.predict_proba(X_data, batch_size=self.batch_size)
            except TypeError:
                y_pred = model.predict_proba(X_data)
            except tf.errors.InternalError:
                X_data = tf.data.Dataset.from_tensor_slices((X_data, y_data)).cache()
                X_data = X_data.batch(batch_size).prefetch(2)
                y_pred = model.predict_proba(X_data)

            try:
                fpr, tpr, thresholds = metrics.roc_curve(y_data, y_pred)
            except ValueError:
                fpr, tpr, thresholds = metrics.roc_curve(y_data, y_pred[:, 1])
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (area = {auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.legend(loc='best')
        plt.title('ROC curve comparison')


class PlotETS(AbstractPlotClass):

    def __init__(self,  plot_folder: str = '.', plot_name='ets'):
        super().__init__(plot_folder, plot_name)
        # self.y_obs = y_obs
        # self.y_preds = y_preds
        self._ax = None
        self._fig = None
        self.ets = pd.DataFrame(index=['ets'])

    def _plot(self, y_obs, y_preds):
        ets = verif.metric.Ets()
        interval = verif.interval.Interval(.5, np.inf, True, True)
        for name, y_pred in y_preds.items():
            self.ets[name] = ets.compute_from_obs_fcst(y_obs, y_pred[..., -1], interval)
        self._fig = self.ets.plot.bar()
        plt.title('ETS (Range: -1/3 to 1, Perfect score: 1, no skill: 0)')