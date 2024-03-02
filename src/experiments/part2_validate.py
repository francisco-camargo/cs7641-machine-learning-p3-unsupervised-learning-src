'''
Run KMeans and GMM on two data
'''

import os

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

import src.data_prep.read_write_data as io
import src.data_prep.data_prep as data_prep
import src.helper_functions as hf
from src.config import p2v_params

# Step 2 Validation
def run_experiment(problem_list, dim_red_algos, random_state, compute, plot_show):
    experiment_name = 'p2v'
    interim_experiment_name = 'p2e'
    interim_folder_path = os.path.join('.', 'data', 'interim', interim_experiment_name)
    folder_path = os.path.join('.', 'data', 'interim', experiment_name)
    print()
    for problem_name in problem_list:
        print(problem_name)
        x_train, y_train, x_test, y_test = data_prep.get_bare_data(problem_name=problem_name)
        
        for algo in dim_red_algos:
            print(f'\t{algo}')
            optimal_n_components = p2v_params[problem_name][algo]
            
            filename = f'{interim_experiment_name}_{problem_name}_{algo}.csv'
            # df = io.read_data(interim_folder_path, filename)
            
            # PCA
            if algo == 'pca':
                dim_red_object = PCA(
                    n_components=optimal_n_components, 
                    random_state=random_state,
                    )

            # ICA
            if algo == 'ica':
                dim_red_object = FastICA(
                    n_components=optimal_n_components,
                    whiten='unit-variance',
                    random_state=random_state,
                    )
                    
            # RCS
            if algo == 'rca':
                dim_red_object = GaussianRandomProjection(
                    n_components=optimal_n_components,
                    random_state=random_state,
                    )
            
            # RFE
            if algo == 'rfe':
                
                x_train_red = x_train[:,optimal_n_components]
                x_test_red = x_test[:,optimal_n_components]
                clf = LogisticRegression()
                cv = StratifiedKFold(10)
                min_features_to_select = len(optimal_n_components)
                dim_red_object = RFECV(
                    estimator=clf,
                    step=1,
                    cv=cv,
                    scoring="f1_weighted",
                    min_features_to_select=min_features_to_select,
                    n_jobs=-1,
                )
                # rfecv.fit(x_train_new, y_train)
                    
            # Fit the algorithms using training data
            try:
                dim_red_object.fit(x_train)
                x_train_red = dim_red_object.transform(x_train)
                x_test_red  = dim_red_object.transform(x_test)
            except TypeError:
                dim_red_object.fit(x_train_red, y_train)
                
            # Save transformed data
            io.save_numpy_array(
                x_train_red,
                folder_path,
                filename=f'{experiment_name}_{problem_name}_{algo}_x_train.csv')
            io.save_numpy_array(
                x_test_red,
                folder_path,
                filename=f'{experiment_name}_{problem_name}_{algo}_x_test.csv')
            io.save_numpy_array(
                y_train,
                folder_path,
                filename=f'{experiment_name}_{problem_name}_{algo}_y_train.csv')
            io.save_numpy_array(
                y_test,
                folder_path,
                filename=f'{experiment_name}_{problem_name}_{algo}_y_test.csv')
            
            # Dummy Classifier
            bare_train_score, bare_test_score = hf.dummy_classifier(x_train, y_train, x_test, y_test)
            new_train_score, new_test_score = hf.dummy_classifier(x_train_red, y_train, x_test_red, y_test)
            
            print('\t\tBare data')
            print(f'\t\t{bare_train_score:.2f}', f'{bare_test_score:.2f}')
            print('\t\tTransformed data')
            print(f'\t\t{new_train_score:.2f}', f'{new_test_score:.2f}')
            print()
            

            if plot_show:
                # Read Data?
                1