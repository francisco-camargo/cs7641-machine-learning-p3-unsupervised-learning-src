#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 15:56:44 2021

@author: francisco camargo
"""

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

import src.helper_functions as hf

mlp_nickname = 'my_mlp'

def train(
    preprocess_algo,
    X_TV,
    y_TV,
    train_dict,
    random_state,
    ):

    # Initialize learner pipeline
    pipeline = Pipeline([])
    learner = MLPClassifier()
    learner.set_params(random_state=random_state)
    pipeline.steps.append((mlp_nickname, learner))
    
    # Set hard-coded parameters for this experiment
    fixed_param = train_dict[preprocess_algo]
    pipeline.set_params(**hf.pipeline_helper(fixed_param, mlp_nickname))
    
    # Set gridsearch parameters
    param_grid = train_dict['gridsearch']['param_grid']
    param_grid = hf.pipeline_helper(param_grid, mlp_nickname)

    # Fit
    print('\tGridSearch verbose:')
    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring = 'f1_weighted',
        n_jobs  = -1,
        refit   = True,
        cv      = train_dict['gridsearch']['grid_cv'],
        verbose = 1,
        return_train_score = False,
        )
    grid.fit(X=X_TV, y=y_TV)

    # Best params
    print('\tBest hyperparameters:', grid.best_params_)
    print(f'\tBest score: {grid.best_score_:.3f}')

    # return grid, pd.DataFrame(grid.cv_results_)
