##################################
# Part 1

p1e_params = {
    'cluster_range': range(2,354)
}

low_bound, high_bound = 2, 21
iter = range(low_bound, high_bound)
p1e_params = {
    'blobs':    iter,
    'monks':    iter,
    'cancer':   iter,
    'iris':     iter,
    'penguins': iter,
    'wine':     iter,
}

p1v_params = {
    'blobs': {
        'kmeans':   2,
        'gmm':      2,
        },
    'monks': {
        'kmeans':   2,#206,
        'gmm':      2,#211,
        },
    'cancer': {
        'kmeans':   2,
        'gmm':      2,
        },
    'iris': {
        'kmeans':   2,
        'gmm':      2,
        },
    'penguins': {
        'kmeans':   2,
        'gmm':      2,
        'dims': [1,2],
        },
    'wine': {
        'kmeans':   3,
        'gmm':      5,
        'dims': [9,11],
        },
}

##################################
# Part 2

p2e_params = {
    'monks':    range(1, 6+1),
    'cancer':   range(1, 30+1),
    'iris':     range(1, 4+1),
    'penguins': range(1, 4+1),
    'wine':     range(1, 13+1),
}

p2v_params = {
    'monks': {
        'pca':6,
        'ica':6,
        'rca':6,
        'rfe': [True, True, True, True, True, True], # 6
        },
    'cancer': {
        'pca':10,
        'ica':30,
        'rca':28,
        'rfe': [False, True, False, True, False, True, \
            True, True, False, False, True, False, True, \
            True, False, True, False, True, False, True, \
            True, True, True, True, True, False, True, \
            True, True, False], # 19
        },
    'iris': {
        'pca':2,
        'ica':1,
        'rca':4,
        'rfe': [False, False, True, True], # 2
        },
    'penguins': {
        'pca':3,
        'ica':1,
        'rca':4,
        'rfe': [True, True, False, True], # 3
        },
    'wine': {
        'pca':10,
        'ica':13,
        'rca':12,
        'rfe': [True, False, True, True, False, False,\
            True, False, False, True, True, True, True], # 8
        },
}

##################################
# Part 3

# maybe I can just do all this in code
# p3e_params = {
#     'blobs': p2v_params + p1e_params,
#     'iris': {
#         'pca':{'n_components':1},
#         'ica':{'n_components':1},
#         'rca':{'n_components':1},
#         },
# }

p3v_params = {
    'monks': {
        'pca': {
            'kmeans': 2,
            'gmm':    3,
        },
        'ica': {
            'kmeans': 2,
            'gmm':    2,
        },
        'rca': {
            'kmeans': 2,
            'gmm':    2,
        },
        'rfe': {
            'kmeans': 2,
            'gmm':    3,
        },
        },
    'cancer': {
        'pca': {
            'kmeans': 2,
            'gmm':    3,
        },
        'ica': {
            'kmeans': 5,
            'gmm':    3,
        },
        'rca': {
            'kmeans': 2,
            'gmm':    2,
        },
        'rfe': {
            'kmeans': 2,
            'gmm':    2,
        },
        },
    'iris': {
        'pca': {
            'kmeans': 29,
            'gmm':    29,
        },
        'ica': {
            'kmeans': 29,
            'gmm':    28,
        },
        'rca': {
            'kmeans': 16,
            'gmm':    16,
        },
        'rfe': {
            'kmeans': 29,
            'gmm':    25,
        },
        },
    'penguins': {
        'pca': {
            'kmeans': 2,
            'gmm':    2,
            'dims': [0,1],
        },
        'ica': {
            'kmeans': 2,
            'gmm':    2,
            'dims': [],
        },
        'rca': {
            'kmeans': 2,
            'gmm':    2,
            'dims': [1,3],
        },
        'rfe': {
            'kmeans': 2,
            'gmm':    2,
            'dims': [1,2],
        },
        },
    'wine': {
        'pca': {
            'kmeans': 3,
            'gmm':    3,
            'dims': [0,1],
        },
        'ica': {
            'kmeans': 19,
            'gmm':    2,
            'dims': [0,11],
        },
        'rca': {
            'kmeans': 2,
            'gmm':    2,
            'dims': [4,5],
        },
        'rfe': {
            'kmeans': 3,
            'gmm':    3,
            'dims': [0,6],
        },
        },
}