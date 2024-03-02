'''

'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from sklearn.metrics import calinski_harabasz_score

import src.data_prep.read_write_data as io
import src.data_prep.data_prep as data_prep
import src.clustering.kmeans as kmeans
import src.clustering.em_gmm as em_gmm
import src.clustering.clustering_evaluation as clustering_evaluation
from src.config import p1e_params

fontsize = 9
fontsize_ticks = fontsize - 2
fig_dim_x = 3.2
fig_dim_y = fig_dim_x * 0.75
alpha = 0.2
linestyles = ['--', '--', '--', '-']
                
# Step 1 exploration
def run_experiment(problem_list, clustering_algos, random_state, compute, plot_show):
    experiment_name = 'p1e'
    interim_folder_path = os.path.join('.', 'data', 'interim', experiment_name)
    
    fig = plt.figure()
    fig.set_size_inches(fig_dim_x, fig_dim_y)
    
    for problem_name in problem_list:
        
        if compute:
            cluster_range = p1e_params[problem_name]
            x_train, y_train, x_test, y_test = data_prep.get_bare_data(problem_name=problem_name)
        
        for algo in clustering_algos:
            filename = f'{experiment_name}_{problem_name}_{algo}.csv'
            
            if compute:
                n_clusters_list = []
                objective_function_list = []
                silhouette_list = []
                mutual_info_train = []
                mutual_info_test = []
                calinski_harabasz = []

                for n_clusters in cluster_range:
                    n_clusters_list.append(n_clusters)
                    
                    # Kmeans
                    if algo == 'kmeans':
                        clusterer, objective_function, cluster_labels, centers = kmeans.my_kmeans(
                            X=x_train, 
                            n_clusters=n_clusters,
                            random_state=random_state)
                        objective_function_list.append(objective_function)

                    # EM GMM
                        # https://towardsdatascience.com/gaussian-mixture-models-and-expectation-maximization-a-full-explanation-50fa94111ddd
                    if algo == 'gmm':
                        clusterer, cluster_labels, centers = em_gmm.em_gmm(
                            X=x_train,
                            n_clusters=n_clusters,
                            random_state=random_state)
                        
                        # if plot_show:
                        #     em_gmm.plot_gmm(
                        #         X=x_train,
                        #         cluster_labels=cluster_labels,
                        #         gmm=clusterer,
                        #         n_clusters=n_clusters,
                        #         x_dim_0=0,
                        #         x_dim_1=1)
                    
                    silhouette_avg, sample_silhouette_values = clustering_evaluation.my_silhouette(
                        X=x_train,
                        n_clusters=n_clusters,
                        cluster_labels=cluster_labels
                        )
                    silhouette_list.append(silhouette_avg)

                    # mutual information
                    mutual_info_train.append(mutual_info_score(y_train, cluster_labels))
                    mutual_info_test.append(mutual_info_score(y_test, clusterer.predict(x_test)))

                    # calinski_harabasz_score
                    calinski_harabasz.append(calinski_harabasz_score(x_train, cluster_labels))

                # Collect Results
                df = pd.DataFrame()
                df['n_clusters']        = n_clusters_list
                df['problem_name']      = problem_name
                df['clustering_algo']   = algo
                df['silhouette']        = silhouette_list
                if algo == 'kmeans':
                    df['kmeansInertia']    = objective_function_list
                df['mutual_info_train'] = mutual_info_train
                df['mutual_info_test']  = mutual_info_test
                df['calinski_harabasz'] = calinski_harabasz

                # Save results
                io.save_data(df, interim_folder_path, filename)
                del df
            
            # Plotting
            if plot_show:
                
                df = io.read_data(interim_folder_path, filename)

                independent_variable = 'n_clusters'
                x = df[independent_variable]
                ylim = [0,0.6]
                
                # Silhouette
                dependent_variable = 'silhouette'
                y = df[dependent_variable]
                y_max_idx = np.argmax(y)
                x_max = x[y_max_idx]
                
                p = plt.plot(x, y, label=f'{problem_name} {algo}')
                color = p[0].get_color() # get str value of color
                ymin, ymax = plt.gca().get_ylim()
                
                plt.vlines(x_max, ylim[0], ylim[1], color=color, linestyles=linestyles.pop()) #, label=f'{problem_name} {algo} peak')
    plt.xlim([2,20])
    plt.xticks([2,5,10,15,20])
    plt.ylim(ylim)
    plt.xlabel('Number of Clusters', fontsize=fontsize)
    plt.ylabel('Silhouette Score', fontsize=fontsize)
    plt.legend(fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.tick_params(direction='in', which='both')
    # plt.title(f'{algo} used on {problem_name} problem')
    plt.tight_layout(pad=0)
    
    # filename = f'{experiment_name}_{problem_name}_{algo}_{dependent_variable}_vs_{independent_variable}'
    filename = f'part1_explore'
    io.save_fig(experiment_name, filename)
    