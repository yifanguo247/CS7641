import os

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA

import experiments


class ICAExperiment(experiments.BaseExperiment):

    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose # False
        self._nn_arch = [(50, 50), (50,), (25,), (25, 25), (100, 25, 100)] # add (2, ) ???
        self._nn_reg = [10 ** -x for x in range(1, 5)] # [0.1, 0.01, 0.001, 0.0001] aka learning rate
        self._clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
        self._dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

    def experiment_name(self):
        return 'ICA'

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-3/blob/master/ICA.py
        self.log("Performing {}".format(self.experiment_name()))

        # %% Data for 1
        ica = FastICA(random_state=self._details.seed) # google how this is done
        kurt = {}
        for dim in self._dims:
            ica.set_params(n_components=dim)
            tmp = ica.fit_transform(self._details.ds.training_x) # performing ICA on training data --> data with dim components
            tmp = pd.DataFrame(tmp)
            tmp = tmp.kurt(axis=0) # calculates fourth moment for each component (# components == dim)
            kurt[dim] = tmp.abs().mean() # average kurtosis among the components, why do this???

        kurt = pd.Series(kurt)
        kurt.to_csv(self._out.format('{}_scree.csv'.format(self._details.ds_name))) # {dim: avg kurtosis of dim components} --> './output/ICA/bank_scree.csv'
        # i assume plotting will go through this .csv file and find the dim that results in highest kurtosis? (what transformation is considered best?)

        # %% Data for 2
        # learn NN on transformed input features, pick the best NN and best feature transformation
        grid = {'ica__n_components': self._dims, 'NN__alpha': self._nn_reg, 'NN__hidden_layer_sizes': self._nn_arch}
        ica = FastICA(random_state=self._details.seed)
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=self._details.seed) # change to 'identity'???
        pipe = Pipeline([('ica', ica), ('NN', mlp)], memory=experiments.pipeline_memory)
        gs, final_estimator = self.gs_with_best_estimator(pipe, grid) # WAIT, best_estimator != final_estimator
        self.log("Grid search complete")

        tmp = pd.DataFrame(gs.cv_results_) # what is this?
        tmp.to_csv(self._out.format('{}_dim_red.csv'.format(self._details.ds_name))) # --> './output/ICA/bank_ICA.csv
        self.log("Done")

    def perform_cluster(self, dim_param):
        self.log('Running clustering for {} with dim param {}'.format(self.experiment_name(), dim_param))
        # TODO: USE UNSUPERVISED METHOD TO GET THIS BEST VALUE
        # %% Data for 3
        # Set this from chart 2 and dump, use clustering script to finish up
        ica = FastICA(n_components=dim_param, random_state=self._details.seed) # user requested dim

        # ANN based on best params from assignment 1
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=self._details.seed)
        pipe = Pipeline([('ica', ica), ('NN', mlp)], memory=experiments.pipeline_memory)
        gs, _ = self.gs_with_best_estimator(pipe, experiments.BEST_NN_PARAMS, type='ass1') # perform experiments and write best estimator params to file

        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(self._out.format('{}_ass1_dim_red.csv'.format(self._details.ds_name))) # write assignment 1 best NN results --> ./output/ICA/bank_ass1_ICA.csv (misleading name. should be ICA)

        hdf_path = self.dump_for_clustering(lambda x: ica.fit_transform(x.get_details().ds.training_x)) # write ICA transformed features + class to file, returns path to that file

        # Run clustering as a subexperiment
        self.log("Running clustering sub-experiment")
        updated_ds = self._details.ds.reload_from_hdf(hdf_path=hdf_path, hdf_ds_name=self._details.ds_name,
                                                      preprocess=False) # new BankData object
        experiments.run_subexperiment(self, self._out.format('clustering/'), updated_ds) # --> './output/ICA/clustering/'
