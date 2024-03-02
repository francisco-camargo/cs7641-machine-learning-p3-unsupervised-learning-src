#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 16:00:24 2021

Create learning curve and print test performance

@author: francisco camargo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve

import src.helper_functions as hf
import src.data_prep.read_write_data as io
from src.mlp.plot_learning_curve import plot_learning_curve

mlp_nickname = 'my_mlp'

def test(
    experiment_name,
    problem_name,
    preprocess_algo,
    X_TV,
    y_TV,
    X_Test,
    y_Test,
    train_dict,
    random_state,
    compute,
    plot_show,
    ):
    
    if compute:
        # Initialize learner pipeline
        pipeline = Pipeline([])
        learner = MLPClassifier()
        learner.set_params(random_state=random_state)
        pipeline.steps.append((mlp_nickname, learner))
        
        # Run Learning Curve and Fit Times Curve
        
        # Set hard-coded parameters for this experiment
        fixed_param = train_dict[preprocess_algo]
        pipeline.set_params(**hf.pipeline_helper(fixed_param, mlp_nickname))

        # Run Learning Curve experiments
        train_sizes, train_scores, validation_scores, fit_times, _ = \
            learning_curve(estimator=pipeline,
                        X=X_TV,
                        y=y_TV,
                        cv=5,
                        scoring='f1_weighted',
                        n_jobs=-1,
                        train_sizes=np.linspace(0.1, 1.0, 20),
                        return_times=True,
                        random_state=random_state)
        train_scores_mean   = np.mean(train_scores, axis=1)
        train_scores_std    = np.std(train_scores, axis=1)
        validation_scores_mean    = np.mean(validation_scores, axis=1)
        validation_scores_std     = np.std(validation_scores, axis=1)
        fit_times_mean      = np.mean(fit_times, axis=1)
        fit_times_std       = np.std(fit_times, axis=1)
        
        # Save data
        df = pd.DataFrame()
        df['train_sizes'] = train_sizes
        df['problem_name'] = problem_name
        df['preprocess_algo'] = preprocess_algo
        df['train_scores_mean'] = train_scores_mean
        df['train_scores_std'] = train_scores_std
        df['validation_scores_mean'] = validation_scores_mean
        df['validation_scores_std'] = validation_scores_std
        df['fit_times_mean'] = fit_times_mean
        df['fit_times_std'] = fit_times_std
        df.set_index(['problem_name', 'preprocess_algo'], inplace=True)

        filename = f'{experiment_name}_{problem_name}_{preprocess_algo}.csv'
        io.save_data_interim(df=df, experiment_name=experiment_name, filename=filename)
        
        # Training Score
        train_mean  = train_scores_mean[-1]
        train_std   = train_scores_std[-1]
        print(f'\t\tTrain score: {train_mean:.3f} +/- {train_std:.3f}')
        
        # Validation Score
        validation_mean  = validation_scores_mean[-1]
        validation_std   = validation_scores_std[-1]
        print(f'\t\tValidation score: {validation_mean:.3f} +/- {validation_std:.3f}')

        # Test model
        final_model = pipeline.fit(X_TV, y_TV)
        predictions = final_model.predict(X_Test)
        result = classification_report(y_Test, predictions, digits=3,output_dict=True)['weighted avg']['f1-score']
        print(f'\t\tTest score: {result:.3f}')
        plt.figure()
        plt.plot(final_model[mlp_nickname].loss_curve_)
    
    if plot_show:
        plot_learning_curve(df, problem_name, preprocess_algo)
        