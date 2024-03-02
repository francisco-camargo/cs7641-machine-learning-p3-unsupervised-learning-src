"""
Pick out a data set and prep it to be used to train a model

"""

import os
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_blobs

import src.data_prep.read_write_data as io

random_state = 42
test_size = 0.2

def get_data_post_feature_engineering(experiment_name, problem_name, algo):
    folder_path = os.path.join('.', 'data', 'interim', experiment_name)
    x_train = io.read_data(folder_path, filename=f'{experiment_name}_{problem_name}_{algo}_x_train.csv', header=None)
    x_train = np.array(x_train).squeeze()
    x_test  = io.read_data(folder_path, filename=f'{experiment_name}_{problem_name}_{algo}_x_test.csv', header=None)
    x_test  = np.array(x_test).squeeze()
    y_train = io.read_data(folder_path, filename=f'{experiment_name}_{problem_name}_{algo}_y_train.csv', dtype=int, header=None)
    y_train = np.array(y_train).squeeze()
    y_test  = io.read_data(folder_path, filename=f'{experiment_name}_{problem_name}_{algo}_y_test.csv', dtype=int, header=None)
    y_test  = np.array(y_test).squeeze()
    return x_train, y_train, x_test, y_test

def sub_prep(df_train, df_test, target_label:str='y'):
    # Data prep
    y_train = df_train[target_label].copy()
    x_train = df_train.drop(columns=[target_label])

    y_test = df_test[target_label].copy()
    x_test = df_test.drop(columns=[target_label])

    # Encode binary target class... not sure if/how this makes any difference
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)
    
    return x_train, y_train, x_test, y_test
    

def get_bare_data(problem_name):
    
    if problem_name == 'monks':
        # Import data
        train_pkl = 'data/interim/monk_interim_train.pkl'
        test_pkl  = 'data/interim/monk_interim_test.pkl'
        df_train, df_test = pd.read_pickle(train_pkl), pd.read_pickle(test_pkl)

        x_train, y_train, x_test, y_test = sub_prep(df_train, df_test, target_label='y')
    
    elif problem_name == 'cancer':
        # Import data
        train_pkl = 'data/interim/cancer_interim_train.pkl'
        test_pkl  = 'data/interim/cancer_interim_test.pkl'
        df_train, df_test = pd.read_pickle(train_pkl), pd.read_pickle(test_pkl)

        x_train, y_train, x_test, y_test = sub_prep(df_train, df_test, target_label='y')
    
    elif problem_name == 'blobs':
        # Generating the sample data from make_blobs
        # This particular setting has one distinct cluster and 3 clusters placed close
        # together.
        x, y = make_blobs(
            n_samples=500,
            n_features=2,
            centers=4,
            cluster_std=1,
            center_box=(-10.0, 10.0),
            shuffle=True,
            random_state=1,
        )  # For reproducibility
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    elif problem_name == 'iris':
        data = datasets.load_iris()
        x, y = data.data, data.target
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    elif problem_name == 'penguins':
        path_train = os.path.join('.', 'data', 'interim', 'penguins_train.pkl')
        path_test  = os.path.join('.', 'data', 'interim', 'penguins_test.pkl')
        df_train, df_test = pd.read_pickle(path_train), pd.read_pickle(path_test)
        
        target_label = 'Species'
        x_train, y_train, x_test, y_test = sub_prep(df_train, df_test, target_label=target_label)
        
    elif problem_name == 'wine':
        data = datasets.load_wine()
        x, y = data.data, data.target
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    # Standard Scaler
    scaler_cred = StandardScaler()
    scaler_cred.fit(x_train)
    x_train = scaler_cred.transform(x_train)
    x_test = scaler_cred.transform(x_test)
    
    return x_train, y_train, x_test, y_test