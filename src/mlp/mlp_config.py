import numpy as np

monks_mlp_config = {
    'vanilla':{
      'hidden_layer_sizes': [4], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.01, # <--
      'max_iter': 5000,
        },
    'kmeans':{
      'hidden_layer_sizes': [3], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.001, # <--
      'max_iter': 5000,
        },
    'gmm':{
      'hidden_layer_sizes': [3], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.001, # <--
      'max_iter': 5000,
        },
    'pca': {
      'hidden_layer_sizes': [4], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.001, # <--
      'max_iter': 5000,
        },
    'ica':{
      'hidden_layer_sizes': [4], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.032, # <--
      'max_iter': 5000,
        },
    'rca':{
      'hidden_layer_sizes': [5,5], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.032, # <--
      'max_iter': 5000,
        },
    'rfe':{
      'hidden_layer_sizes': [4], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.01, # <--
      'max_iter': 5000,
        },
    'gridsearch': {
        'grid_cv': 5,
        'param_grid':{
            'hidden_layer_sizes': [[2], [3], [4], [5], [1,1], [2,2], [3,3], [4,4], [5,5]],
            'learning_rate_init' : np.logspace(-4, -1.5, 6),
            }
        }
    }

cancer_mlp_config = {
    'vanilla':{
      'hidden_layer_sizes': [2,2], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.0032, # <--
      'max_iter': 5000,
        },
    'kmeans':{
      'hidden_layer_sizes': [3], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.0032, # <--
      'max_iter': 5000,
        },
    'gmm':{
      'hidden_layer_sizes': [3], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.032, # <--
      'max_iter': 5000,
        },
    'pca': {
      'hidden_layer_sizes': [2], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.001, # <--
      'max_iter': 5000,
        },
    'ica':{
      'hidden_layer_sizes': [], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.001, # <--
      'max_iter': 5000,
        },
    'rca':{
      'hidden_layer_sizes': [], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.0032, # <--
      'max_iter': 5000,
        },
    'rfe':{
      'hidden_layer_sizes': [2,2], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.00032, # <--
      'max_iter': 5000,
        },
    'gridsearch': {
        'grid_cv': 5,
        'param_grid':{
            'hidden_layer_sizes': [[], [1], [2], [3], [1,1], [2,2], [3,3]],
            'learning_rate_init' : np.logspace(-4, -1.5, 6),
            }
        }
    }

iris_mlp_config = {
    'vanilla':{
      'hidden_layer_sizes': [1], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.00316, # <--
      'max_iter': 5000,
        },
    'kmeans':{
      'hidden_layer_sizes': [5], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.000316, # <--
      'max_iter': 5000,
        },
    'gmm':{
      'hidden_layer_sizes': [5], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.000316, # <--
      'max_iter': 5000,
        },
    'pca': {
      'hidden_layer_sizes': [3,3], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.0316, # <--
      'max_iter': 5000,
        },
    'ica':{
      'hidden_layer_sizes': [5], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.0001, # <--
      'max_iter': 5000,
        },
    'rca':{
      'hidden_layer_sizes': [5], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.0001, # <--
      'max_iter': 5000,
        },
    'rfe':{
      'hidden_layer_sizes': [5], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.01, # <--
      'max_iter': 5000,
        },
    'gridsearch': {
        'grid_cv': 5,
        'param_grid':{
            'hidden_layer_sizes': [[], [1], [2], [3], [4], [5], [1,1], [2,2], [3,3], [4,4], [5,5]],
            'learning_rate_init' : np.logspace(-4, -1.5, 6),
            }
        }
    }

penguins_mlp_config = {
    'vanilla':{
      'hidden_layer_sizes': [2], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.032, # <--
      'max_iter': 5000,
        },
    'kmeans':{
      'hidden_layer_sizes': [5,5], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.00316, # <--
      'max_iter': 5000,
        },
    'gmm':{
      'hidden_layer_sizes': [5,5], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.00316, # <--
      'max_iter': 5000,
        },
    'pca': {
      'hidden_layer_sizes': [5], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.00316, # <--
      'max_iter': 5000,
        },
    'ica':{
      'hidden_layer_sizes': [5,5], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.0032, # <--
      'max_iter': 5000,
        },
    'rca':{
      'hidden_layer_sizes': [2,2], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.0316, # <--
      'max_iter': 5000,
        },
    'rfe':{
      'hidden_layer_sizes': [2], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.00316, # <--
      'max_iter': 5000,
        },
    'gridsearch': {
        'grid_cv': 5,
        'param_grid':{
            'hidden_layer_sizes': [[], [1], [2], [3], [4], [5], [1,1], [2,2], [3,3], [4,4], [5,5]],
            'learning_rate_init' : np.logspace(-4, -1.5, 6),
            }
        }
    }

wine_mlp_config = {
    'vanilla':{
      'hidden_layer_sizes': [3], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.001, # <--
      'max_iter': 5000,
        },
    'kmeans':{
      'hidden_layer_sizes': [3], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.0316, # <--
      'max_iter': 5000,
        },
    'gmm':{
      'hidden_layer_sizes': [5], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.01, # <--
      'max_iter': 5000,
        },
    'pca': {
      'hidden_layer_sizes': [5,5], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.00316, # <--
      'max_iter': 5000,
        },
    'ica':{
      'hidden_layer_sizes': [], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.00316, # <--
      'max_iter': 5000,
        },
    'rca':{
      'hidden_layer_sizes': [5,5], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.01, # <--
      'max_iter': 5000,
        },
    'rfe':{
      'hidden_layer_sizes': [3], # <--
      'activation': 'relu',
      'solver': 'adam',
      'alpha': 0.0035,
      'batch_size': 'auto',
      'learning_rate': 'constant', 
      'learning_rate_init' : 0.001, # <--
      'max_iter': 5000,
        },
    'gridsearch': {
        'grid_cv': 5,
        'param_grid':{
            'hidden_layer_sizes': [[], [1], [2], [3], [4], [5], [1,1], [2,2], [3,3], [4,4], [5,5]],
            'learning_rate_init' : np.logspace(-4, -1.5, 6),
            }
        }
    }