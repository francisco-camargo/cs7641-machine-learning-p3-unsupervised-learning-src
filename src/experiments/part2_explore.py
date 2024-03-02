# PCA theory
    # https://towardsdatascience.com/principal-component-analysis-part-1-the-different-formulations-6508f63a5553
    
# choosing a cutoff
    # https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/ 
    
# LDA, Isomap, Locally Linear Embedding
    # https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python/
    # UMAP https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris/notebook#UMAP
    # https://stackabuse.com/dimensionality-reduction-in-python-with-scikit-learn/

# https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python/

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from src.config import p2e_params
import src.data_prep.data_prep as data_prep
import src.helper_functions as hf
import src.data_prep.read_write_data as io

experiment_name = 'p2e'

# Step 2 exploration
def run_experiment(problem_list, dim_red_algos, random_state):
    
    for problem_name in problem_list:
        print()
        print(problem_name)
        
        n_components_range = p2e_params[problem_name]
        x_train, y_train, x_test, y_test = data_prep.get_bare_data(problem_name=problem_name)

        for algo in dim_red_algos:
            filename = f'{experiment_name}_{problem_name}_{algo}.csv'
            print(f'\t{algo}')
            
            if algo == 'pca':
                # pick number of components by grabbing enough components to exceed variance threshold
                # validate by
                pca_threshold = 0.95
                pca = PCA(n_components=list(n_components_range)[-1], random_state=random_state)
                pca.fit(x_train)
                eigenvalues = pca.explained_variance_ratio_
                explained_variance = pca.explained_variance_ratio_
                
                num_keep_components, cumsum_kept = hf.cumsum_threshold_filter(explained_variance, pca_threshold)
                print(f'{problem_name} {algo} keep {num_keep_components} features to capture {cumsum_kept:.2f} variance')
                
                pca2 = PCA(n_components=num_keep_components, random_state=random_state)
                r2 = hf.reconstruction_error(x_train=x_train, learner=pca2)
                print(f'\tReconstruction Error: {r2:.2f}')
                
                df = pd.DataFrame()
                df['n_components'] = list(n_components_range)
                df['problem_name'] = problem_name
                df['algo'] = algo
                df['eigenvalues'] = eigenvalues
                df['explained_variance'] = explained_variance
                df['r2'] = np.cumsum(explained_variance)
                df['threshold'] = pca_threshold
                df['num_keep_components'] = num_keep_components
                df['cumsum_kept'] = cumsum_kept
                df.set_index(['problem_name', 'algo'], inplace=True)
                
                io.save_data_interim(df=df, experiment_name=experiment_name, filename=filename)
                
                
            if algo == 'ica':
                # https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_vs_pca.html#sphx-glr-auto-examples-decomposition-plot-ica-vs-pca-py
                # https://edstem.org/us/courses/32923/discussion/2767458
                # look at kurtosis!
                kurt_list = []
                r2_list = []
                
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                        # Keep getting FastICA convergences warnings. I'm not going to do anything about them, so let's suppress them
                    for n_components in n_components_range:
                        ica = FastICA(
                            n_components=n_components,
                            whiten='unit-variance',
                            random_state=random_state,
                            )
                        ica.fit(x_train)
                        new_x_train = ica.transform(x_train)
                        avg_kurtosis = hf.calc_kurtosis(new_x_train)
                        kurt_list.append(avg_kurtosis)

                        r2_list.append(hf.reconstruction_error(x_train=x_train, learner=ica))
                        
                    peak_kurt_n_components = np.argmax(kurt_list)+1
                    
                df = pd.DataFrame()
                df['n_components'] = list(n_components_range)
                df['problem_name'] = problem_name
                df['algo'] = algo
                df['kurtosis'] = kurt_list
                df['r2'] = r2_list
                df['num_keep_components'] = peak_kurt_n_components
                df.set_index(['problem_name', 'algo'], inplace=True)
                
                io.save_data_interim(df=df, experiment_name=experiment_name, filename=filename)
                
                # for idx, item in enumerate(zip(kurt_list, r2_list)):
                #     print(idx+1, f'{item[0]:.2f}', f'{item[1]:.2f}')
                # print()
                # print(peak_kurt_n_components-1, kurt_list[peak_kurt_n_components-1], r2_list[peak_kurt_n_components-1])
            
            if algo == 'rca':
                r2_list = []
                
                for n_components in n_components_range:
                    local_r2 = []
                    for random_state_local in range(10):
                        rca = GaussianRandomProjection(
                            n_components=n_components,
                            random_state=random_state_local,
                            )
                        # rca = SparseRandomProjection(
                        #     n_components=n_components,
                        #     random_state=random_state_local,
                        #     )
                        rca.fit(x_train)
                        local_r2.append(hf.reconstruction_error(x_train=x_train, learner=rca))
                    r2_list.append(np.mean(local_r2))
                    # new_x_train = rca.transform(x_train)
                
                # pick best n_components
                rca_threshold = 0.90
                num_keep_components, r2_kept = hf.threshold_filter(r2_list, rca_threshold)
                
                # plt.figure()
                # plt.plot(n_components_range, r2_list)
                # plt.vlines(num_keep_components, 0, 1)
                # plt.title(f'{problem_name} solved with {algo} with r2 of {r2_kept:.2f}')
                
                df = pd.DataFrame()
                df['n_components'] = list(n_components_range)
                df['problem_name'] = problem_name
                df['algo'] = algo
                df['r2'] = r2_list
                df['threshold'] = rca_threshold
                df['num_keep_components'] = num_keep_components
                df['cumsum_kept'] = r2_kept
                df.set_index(['problem_name', 'algo'], inplace=True)
                
                io.save_data_interim(df=df, experiment_name=experiment_name, filename=filename)
                
            if algo == 'rfe':
                # https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py
                
                clf = LogisticRegression()
                cv = StratifiedKFold(10)

                # Base model
                min_features_to_select = x_train.shape[1]  # Minimum number of features to consider
                rfecv = RFECV(
                    estimator=clf,
                    step=1,
                    cv=cv,
                    scoring="f1_weighted",
                    min_features_to_select=min_features_to_select,
                    n_jobs=-1,
                )
                rfecv.fit(x_train, y_train)
                keep_col_list = rfecv.get_support()
                
                bare_train_score, bare_test_score = hf.my_f1_score(x_train, y_train, x_test, y_test, rfecv)
                
                print(f"\tBase number of features: {rfecv.n_features_}")
                print(f'\t\tTrain score: {bare_train_score:.2f}')
                print(f'\t\tTest score: {bare_test_score:.2f}')
                
                # with RFE
                min_features_to_select = 1  # Minimum number of features to consider
                rfecv = RFECV(
                    estimator=clf,
                    step=1,
                    cv=cv,
                    scoring="f1_weighted",
                    min_features_to_select=min_features_to_select,
                    n_jobs=-1,
                )
                rfecv.fit(x_train, y_train)
                keep_col_list = rfecv.get_support()
                
                train_score, test_score = hf.my_f1_score(x_train, y_train, x_test, y_test, rfecv)

                print()
                print(f"\tOptimal number of features: {rfecv.n_features_}")
                print(f'\t\tTrain score: {train_score:.2f}')
                print(f'\t\tTest score: {test_score:.2f}')
                print(f'\t\tColumns to keep: {keep_col_list}')

                # Save data
                df = pd.DataFrame()
                df['n_components'] = list(n_components_range)
                df['problem_name'] = problem_name
                df['algo'] = algo
                df['keep_boolean'] = keep_col_list
                df['num_keep_components'] = rfecv.n_features_
                df['bare_train_score'] = bare_train_score
                df['bare_test_score'] = bare_test_score
                df['train_score'] = train_score
                df['test_score'] = test_score
                df.set_index(['problem_name', 'algo'], inplace=True)
                
                io.save_data_interim(df=df, experiment_name=experiment_name, filename=filename)
                
fontsize = 9
fontsize_ticks = fontsize - 2
fig_dim_x = 3.2
fig_dim_y = fig_dim_x * 0.75
alpha = 0.2

def plot_experiment(problem_list, dim_red_algos):
    
    for algo in dim_red_algos:
        fig = plt.figure()
        fig.set_size_inches(fig_dim_x, fig_dim_y)
        for problem_name in problem_list:
            filename = f'{experiment_name}_{problem_name}_{algo}.csv'
            folder_path = os.path.join('.', 'data', 'interim', experiment_name)
            df = io.read_data(folder_path, filename)
            mask = (df['algo']==algo) & (df['problem_name']==problem_name)
            df = df[mask]
            
            if algo == 'pca':
                plt.plot(df['n_components'], df['r2'], label=f'{problem_name} {algo}')
                plt.xlim([1,13])
                # plt.xticks(np.arange(0, 16, step=4))
                plt.xticks([1,5,10,13])
                plt.ylim([0.3, 1])
                plt.xlabel('Number of Components', fontsize=fontsize)
                plt.ylabel('Cumulative Variance', fontsize=fontsize)
                plt.axhline(y=df['threshold'][0], color='k', xmin=0, xmax=13, linestyle='--')
                
            if algo == 'ica':
                plt.plot(df['n_components'], df['kurtosis'], label=f'{problem_name} {algo}')
                plt.xlim([1,13])
                # plt.xticks(np.arange(0, 16, step=4))
                plt.xticks([1,5,10,13])
                # plt.ylim([0, 1])
                plt.xlabel('Number of Components', fontsize=fontsize)
                plt.ylabel('Kurtosis', fontsize=fontsize)
                # plt.axvline(x=df['num_keep_components'][0], color='k', ymin=0, ymax=1, linestyle='--')
            
            if algo == 'rca':
                plt.plot(df['n_components'], df['r2'], label=f'{problem_name} {algo}')
                plt.xlim([1,13])
                # plt.xticks(np.arange(0, 16, step=4))
                plt.xticks([1,5,10,13])
                plt.ylim([0, 1])
                plt.xlabel('Number of Components', fontsize=fontsize)
                plt.ylabel('Reconstruction Error', fontsize=fontsize)
                plt.axhline(y=df['threshold'][0], color='k', xmin=0, xmax=13, linestyle='--')
                
            if algo == 'rfe':
                1
                

        plt.legend(fontsize=fontsize_ticks)
        plt.xticks(fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)
        plt.tick_params(direction='in', which='both')
        # plt.title(f'{algo} used on {problem_name} problem')
        plt.tight_layout(pad=0)
        filename = f'{experiment_name}_{algo}'
        io.save_fig(experiment_name, filename)