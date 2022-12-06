import logging
import copy
import os

import dask
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client

from dask_ml.model_selection import train_test_split, HyperbandSearchCV
from dask_ml.preprocessing import DummyEncoder


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import datasets, metrics, model_selection, svm
from sklearn.utils import class_weight
# from scikeras.wrappers import KerasClassifier
from scipy.stats import loguniform, uniform

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow_io as tfio

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import verif
from pathlib import Path


from citibike.plotting.abstract_plot_class import AbstractPlotClass
from citibike.helpers import TimeTrackingWrapper, replace_space, haversine_dist, to_list, remove_items, standardize

import warnings

import lightgbm as ltb


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

    Requires models to provide `predict_proba()` method!
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


class DateTimeSplitter:
    @staticmethod
    def to_attributes(data, time_split_attributes, original_time_name='starttime', drop_original_time_name=False,
                      return_as_categorized=False, return_used_attributes=False):
        _time_mapping = {'start hour': data[original_time_name].dt.hour,
                         'start day of week': data[original_time_name].dt.dayofweek,
                         'start day name': data[original_time_name].dt.day_name(),
                         'start month': data[original_time_name].dt.month,
                         }
        res = data[[original_time_name]]
        time_split_attributes = to_list(time_split_attributes)
        used_attributes = []
        for var in time_split_attributes:
            if var in _time_mapping.keys():
                res[var] = _time_mapping[var]
                used_attributes.append(var)

        if drop_original_time_name:
            cols = remove_items(list(res.columns), original_time_name)
            res = res[cols]

        if return_as_categorized:
            res = res.categorize(time_split_attributes)

        if return_used_attributes:
            res = (res, used_attributes)

        return res


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


def get_keras_model(input_shape=(None, 49), drop_rate=.25, compile_opts=None):
    model = Sequential([
        Dense(512, activation='relu', input_shape=input_shape),
        Dropout(rate=drop_rate),
        Dense(256, activation='relu'),
        Dropout(rate=drop_rate),
        Dense(1, activation='sigmoid')
    ])
    # opt = tf.keras.optimizers.SGD(lr=lr, momentum=momentum)
    # tf.keras.optimizers.Adam(1e-3)
    # learning_rate = lr, momentum = momentum, nesterov = True,
    if compile_opts is None:
        compile_opts = {'loss': "binary_crossentropy", 'optimizer': 'adam', 'metrics': ["accuracy", tf.keras.metrics.AUC(name='auc')]}
    model.compile(**compile_opts)
    return model


def make_ds(features, labels, buffer_size, prefetch=False):
    '''
    Make TF Dataset from pd.DataFrames
    :param prefetch:

    '''
    ds = tf.data.Dataset.from_tensor_slices((features, labels)) # .cache()
    ds = ds.shuffle(buffer_size, reshuffle_each_iteration=True).repeat(count=-1)
    if prefetch:
        ds = ds.batch(batch_size).prefetch(2)
    return ds


def make_val_test_ds(X_data, y_data, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((X_data, y_data))#.cache()
    ds = ds.batch(batch_size).prefetch(2)
    return ds


def make_balanced_ds(X_train, y_train, batch_size, buffer_size):
    pos_ds = make_ds(X_train[y_train == 1], y_train[y_train == 1], buffer_size)
    neg_ds = make_ds(X_train[y_train < 1], y_train[y_train < 1], buffer_size)
    ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
    ds = ds.batch(batch_size).prefetch(2)
    return ds


def train_tf_model(train_ds, compile_opts, fit_opts, dropout_rate):
    logging.info(f'`train_tf_model` called with \n compile_opts={compile_opts} \n '
                 f'fit_opts={fit_opts} \n dropout_rate={dropout_rate}')

    modeltf = get_keras_model(input_shape=[X_val.shape[1]], drop_rate=dropout_rate, compile_opts=compile_opts)
    history = modeltf.fit(
        train_ds,
        **fit_opts
        # epochs=10,
        # steps_per_epoch=steps_per_epoch,
        # # callbacks=[early_stopping],
        # validation_data=val_ds
    )

    return modeltf, history


def train_and_make_pred(X_train, y_train, X_test, y_test, compile_opts, fit_opts, batch_size, buffer_size,
                        balanced=True, **kwargs):
    if balanced:
        model, history = train_tf_model(train_ds=make_balanced_ds(X_train, y_train,
                                                                  batch_size=batch_size,
                                                                  buffer_size=buffer_size
                                                                  ), compile_opts=compile_opts, fit_opts=fit_opts,
                                        **kwargs)
    else:
        model, history = train_tf_model(train_ds=make_ds(X_train, y_train, buffer_size=buffer_size, prefetch=True),
                                        compile_opts=compile_opts, fit_opts=fit_opts, **kwargs)

    test_ds = make_val_test_ds(X_test, y_test, batch_size=batch_size)
    y_pred = model.predict_proba(test_ds)
    return y_pred, history


if __name__ == '__main__':

    warnings.simplefilter(action='ignore', category=FutureWarning)
    logging.getLogger().setLevel(logging.INFO)

    plot_list = ['PlotFromAtoBHeatmap', 'PlotDurationDist', 'PlotDurationDist_ext', 'PlotNumRidesPerTimes',
                 'PlotNumRidesPerYearStation', 'PlotROCs', 'PlotETS']
    preprocess = False
    create_plots = True
    preproc_data_path = 'preproc_data_path'
    pred_data_path = 'pred_data_path'

    dtypes = {'tripduration': np.int32,  # starttime and stoptime via parse_dates
              'start station id': pd.Int32Dtype(), 'start station name': 'category',
              'start station latitude': np.float32, 'start station longitude': np.float32,
              'end station id': pd.Int32Dtype(), 'end station name': 'category', 'end station latitude': np.float32,
              'end station longitude': np.float32, 'bikeid': np.int32, #'usertype': 'category',
              'birth year': np.int32, 'gender': np.int16}

    if preprocess:
        client_ = Client()

    if len(plot_list) > 0:
        file_names = [f'NYC_data/2018{i:02}-citibike-tripdata.csv' for i in range(1, 13)]
        ddf_quick = dd.read_csv(file_names, parse_dates=['starttime', 'stoptime'], infer_datetime_format=True,
                                dtype=dtypes)
        ddf_quick = ddf_quick.categorize(['gender', 'usertype', 'end station name', 'start station name'])
        ddf_quick['tripduration'] = ddf_quick['tripduration'] / 60

    if 'PlotFromAtoBHeatmap' in plot_list:

        from_a_to_b = PlotFromAtoBHeatmap(ddf_quick, y_name='start station id',
                                          x_name='end station id', plot_folder='plots')
        from_a_to_b.plot(log_scale=False)
        from_a_to_b.plot(log_scale=True)

    if 'PlotDurationDist' in plot_list:

        sample_overview = PlotDurationDist(ddf_quick, plot_folder='plots',
                                           additional_vars=['start day of week', 'start hour']
                                          )
        sample_overview.plot(n_samples=100000, diag_kws=dict(edgecolor=None))

    if 'PlotDurationDist_ext' in plot_list:

        sample_overview_ext = PlotDurationDist(ddf_quick, plot_folder='plots', plot_name='duration_dist_exp',
                                           additional_vars=['start day of week', 'start hour', 'gender', 'birth year']
                                          )
        sample_overview_ext.plot(n_samples=200000, diag_kws=dict(edgecolor=None))

    if 'PlotNumRidesPerTimes' in plot_list:
        ddf_c = ddf_quick[['starttime', 'gender', 'usertype']]
        time_splits = DateTimeSplitter.to_attributes(ddf_c, time_split_attributes=['start hour', 'start day of week',
                                                                                   'start day name', 'start month'],
                                                     original_time_name='starttime', drop_original_time_name=True)
        ddf_c = dd.concat([ddf_c, time_splits], axis=1)
        ddf_c = ddf_c.persist()

        num_rides_per_time = PlotNumRidesPerXY(data=ddf_c, plot_folder='plots')

        num_rides_per_time.plot(per_col_name='start day of week', category='usertype')
        num_rides_per_time.plot(per_col_name='start day of week', category='gender')

        num_rides_per_time.plot(per_col_name='start month', category='usertype')
        num_rides_per_time.plot(per_col_name='start month', category='gender')

        num_rides_per_time.plot(per_col_name='start hour', category='usertype')
        num_rides_per_time.plot(per_col_name='start hour', category='gender')

    if 'PlotNumRidesPerYearStation' in plot_list:

        num_rides_per = PlotNumRidesPerXY(data=ddf_quick, plot_folder='plots')
        num_rides_per.plot(per_col_name='birth year', category='usertype', min_idx_value=1940)
        num_rides_per.plot(per_col_name='birth year', category='gender', min_idx_value=1940)
        num_rides_per.plot(per_col_name='start station id', category='usertype')
        num_rides_per.plot(per_col_name='start station id', category='gender')

    logging.debug('debug marker')

    if preprocess:

        y = ddf_c[['usertype']].replace('Subscriber', 0).replace('Customer', 1).astype(np.int64) # labels


        x_features = {'start hour', 'start day of week', 'start month', 'birth year', 'gender', 'tripduration'}
        orig_feat = list(set(ddf_quick.columns).intersection(x_features))
        constr_feat = list(x_features - set(ddf_quick.columns))
        X = dd.concat([ddf_quick[orig_feat], ddf_c[constr_feat]], axis=1)

        X['dist'] = haversine_dist(lat1=ddf_quick['start station latitude'].values,
                                   lon1=ddf_quick['start station longitude'].values,
                                   lat2=ddf_quick['end station latitude'].values,
                                   lon2=ddf_quick['end station longitude'].values
                                   )

        time_vars = ['start hour', 'start day of week', 'start month']

        Xt = X[(X['birth year'] >= 1940) & (X['tripduration'] <= 600) & (X['dist'] <= 20)]
        yt = y[(X['birth year'] >= 1940) & (X['tripduration'] <= 600) & (X['dist'] <= 20)]

        Xt_cat = Xt.categorize(time_vars)

        encoder = DummyEncoder()
        X_enc = encoder.fit_transform(Xt_cat)

        # using shuffle as index is no time series
        X_tr_val, X_test, y_tr_val, y_test = train_test_split(X_enc, yt, test_size=.2,
                                                              shuffle=True, random_state=0)
        X_train, X_val, y_train, y_val = train_test_split(X_tr_val, y_tr_val, test_size=.2,
                                                          shuffle=True, random_state=0)

        if not os.path.exists(preproc_data_path):
            os.makedirs(preproc_data_path)
        X_train.to_parquet(f'{preproc_data_path}/X_train.parquet', engine='pyarrow')
        y_train.to_parquet(f'{preproc_data_path}/y_train.parquet', engine='pyarrow')
        X_val.to_parquet(f'{preproc_data_path}/X_val.parquet', engine='pyarrow')
        y_val.to_parquet(f'{preproc_data_path}/y_val.parquet', engine='pyarrow')
        X_test.to_parquet(f'{preproc_data_path}/X_test.parquet', engine='pyarrow')
        y_test.to_parquet(f'{preproc_data_path}/y_test.parquet', engine='pyarrow')

    # client_.restart()
    # X_train = dd.read_parquet(f'{preproc_data_path}/X_train.parquet', engine='pyarrow').to_dask_array(lengths=True)
    # y_train = dd.read_parquet(f'{preproc_data_path}/y_train.parquet', engine='pyarrow').to_dask_array(lengths=True)
    # X_val = dd.read_parquet(f'{preproc_data_path}/X_val.parquet', engine='pyarrow').to_dask_array(lengths=True)
    # y_val = dd.read_parquet(f'{preproc_data_path}/y_val.parquet', engine='pyarrow').to_dask_array(lengths=True)
    # X_test = dd.read_parquet(f'{preproc_data_path}/X_test.parquet', engine='pyarrow').to_dask_array(lengths=True)
    # y_test = dd.read_parquet(f'{preproc_data_path}/y_test.parquet', engine='pyarrow').to_dask_array(lengths=True)
    #
    # y_train = y_train.flatten()
    # y_val = y_val.flatten()
    # y_test = y_test.flatten()

    X_train = pd.read_parquet(f'{preproc_data_path}/X_train.parquet', engine='pyarrow')
    y_train = pd.read_parquet(f'{preproc_data_path}/y_train.parquet', engine='pyarrow')
    X_val = pd.read_parquet(f'{preproc_data_path}/X_val.parquet', engine='pyarrow')
    y_val = pd.read_parquet(f'{preproc_data_path}/y_val.parquet', engine='pyarrow')
    X_test = pd.read_parquet(f'{preproc_data_path}/X_test.parquet', engine='pyarrow')
    y_test = pd.read_parquet(f'{preproc_data_path}/y_test.parquet', engine='pyarrow')

    y_train = y_train.values.flatten()
    y_val = y_val.values.flatten()
    y_test = y_test.values.flatten()

    # standardize some features that are not already one hot encoded
    cols_to_standardize = ['birth year', 'tripduration', 'dist']
    X_train[cols_to_standardize], train_mean, train_std = standardize(X_train[cols_to_standardize])
    X_val[cols_to_standardize], _, _ = standardize(X_val[cols_to_standardize], mean=train_mean, std=train_std)
    X_test[cols_to_standardize], _, _ = standardize(X_test[cols_to_standardize], mean=train_mean, std=train_std)


    logging.debug('debug marker: NN')

    # TF DataSet sizes
    buffer_size = 100000
    batch_size = 1024

    resampled_steps_per_epoch_balanced = int(np.ceil(2.0 * y_train[y_train == 1].shape[0] / batch_size))


    tf.keras.models.Model.predict_proba = tf.keras.models.Model.predict

    compile_opts = {'loss': tf.keras.losses.BinaryCrossentropy(),
                    'optimizer': tf.keras.optimizers.Adam(learning_rate=1e-3),
                    'metrics': ["accuracy", tf.keras.metrics.AUC(name='auc')]}
    fit_opts = {'epochs': 50,
                'validation_data': make_val_test_ds(X_val, y_val, batch_size=batch_size*2),
                'steps_per_epoch': resampled_steps_per_epoch_balanced,
                'validation_steps': 20,
                }
    train_balanced_tf = False
    if train_balanced_tf:
        logging.info('running training for train_balanced_tf')
        y_pred_tf_balanced, hist_tf_balanced = train_and_make_pred(X_train, y_train, X_test, y_test,
                                                                   compile_opts, fit_opts, batch_size, buffer_size,
                                                                   dropout_rate=.25)
        if not os.path.exists(pred_data_path):
            os.makedirs(pred_data_path)
        np.save(f'{pred_data_path}/y_pred_tf_balanced', y_pred_tf_balanced)

    train_class_weights_tf = False
    if train_class_weights_tf:
        logging.info('running training for train_class_weights_tf')
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_val), y=y_val)
        fit_opts['class_weight'] = {0: class_weights[0], 1: class_weights[1]}
        y_pred_tf_class_weights, hist_tf_class_weights = train_and_make_pred(X_train, y_train, X_test, y_test,
                                                                             compile_opts, fit_opts,
                                                                             batch_size, buffer_size, balanced=False
                                                                             )
        if not os.path.exists(pred_data_path):
            os.makedirs(pred_data_path)
        np.save(f'{pred_data_path}/y_pred_tf_class_weights', y_pred_tf_class_weights)

    train_class_weights_tf_drop = False
    if train_class_weights_tf_drop:
        logging.info('running training for train_class_weights_tf_drop')
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_val), y=y_val)
        fit_opts['class_weight'] = {0: class_weights[0], 1: class_weights[1]}

        y_pred_tf_class_weights_drop, hist_tf_class_weights_drop = train_and_make_pred(X_train, y_train, X_test, y_test,
                                                                             compile_opts, fit_opts,
                                                                             batch_size, buffer_size, balanced=False,
                                                                                       dropout_rate=.25
                                                                             )
        if not os.path.exists(pred_data_path):
            os.makedirs(pred_data_path)

        np.save(f'{pred_data_path}/y_pred_tf_class_weights_drop', y_pred_tf_class_weights_drop)

    logging.debug('debug marker: RF and GB')

    train_rfc = False
    if train_rfc:
        logging.info('running training for train_rfc')
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, verbose=4, class_weight='balanced')
        clf.fit(X_val, y_val)
        np.save(f'{pred_data_path}/y_pred_clf', clf.predict_proba(X_test))
        # clf.fit(X_train,y_train)

    train_gbc = False
    if train_gbc:
        logging.info('running training for train_gbc')
        modelgb = GradientBoostingClassifier(n_estimators=50, verbose=4)
        modelgb.fit(X_val, y_val)
        np.save(f'{pred_data_path}/y_pred_modelgb', modelgb.predict_proba(X_test))
        # modelgb.fit(X_train,y_train)


    # modeltf = get_keras_model(input_shape=49, )
    logging.debug('debug marker')
    if preprocess:
        client_.restart()


    logging.debug('debug marker')



    y_preds = {'tf_balanced': np.load(f'{pred_data_path}/y_pred_tf_balanced.npy'),
               'tf_class_weights': np.load(f'{pred_data_path}/y_pred_tf_class_weights.npy'),
               'tf_class_weights_drop': np.load(f'{pred_data_path}/y_pred_tf_class_weights_drop.npy'),
               'clf': np.load(f'{pred_data_path}/y_pred_clf.npy'),
               'gb': np.load(f'{pred_data_path}/y_pred_modelgb.npy')
               }

    if 'PlotROCs' in plot_list:
        rocs = PlotROCs(plot_folder='plots')
        rocs.plot(y_preds=y_preds, y_obs=y_test)

    if 'PlotETS' in plot_list:
        ets = PlotETS(plot_folder='plots')
        ets.plot(y_obs=y_test, y_preds=y_preds)

