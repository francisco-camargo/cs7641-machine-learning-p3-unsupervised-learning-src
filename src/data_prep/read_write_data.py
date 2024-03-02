import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_data(df, folder_path, filename):
    try: os.makedirs(folder_path)
    except FileExistsError: pass # if this folder already exists, no need to remake it
    path = os.path.join(folder_path, filename)
    df.to_csv(path)
    
def save_data_interim(df, experiment_name, filename):
    folder_path = os.path.join('.', 'data', 'interim', experiment_name)
    try: os.makedirs(folder_path)
    except FileExistsError: pass
    save_data(df, folder_path, filename)

def save_data_results(df, experiment_name, filename):
    folder_path = os.path.join('.', 'data', 'results', experiment_name)
    try: os.makedirs(folder_path)
    except FileExistsError: pass
    save_data(df, folder_path, filename)

def save_fig(experiment_name, filename):
    folder_path = os.path.join('data', 'results', experiment_name)
    try: os.makedirs(folder_path)
    except FileExistsError: pass
    for extension in ['.png', '.eps']:
        path = os.path.join(folder_path, filename + extension)
        plt.savefig(path)
        
def save_numpy_array(X, folder_path, filename):
    try: os.makedirs(folder_path)
    except FileExistsError: pass
    filename = os.path.join(folder_path, filename)
    np.savetxt(filename, X, delimiter=",")
    
def read_data(folder_path, filename, header='infer', dtype=None):
    path = os.path.join(folder_path, filename)
    df = pd.read_csv(path, dtype=dtype, header=header)
    return df
