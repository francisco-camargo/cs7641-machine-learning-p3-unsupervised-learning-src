'''
Run KMeans and GMM on two data
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.data_prep.read_write_data as io
import seaborn as sns

import src.data_prep.data_prep as data_prep
import src.clustering.kmeans as kmeans
import src.clustering.em_gmm as em_gmm
import src.clustering.clustering_evaluation as clustering_evaluation
import src.clustering.clustering_plotting as clustering_plotting
from src.config import p1v_params
import src.helper_functions as hf

fontsize = 9
fontsize_ticks = fontsize - 2
fontsize_title = fontsize + 2
fig_dim_x = 3.2
fig_dim_y = fig_dim_x * 0.75
alpha = 0.2

# Step 1 Validation
def run_experiment(problem_list, clustering_algos, random_state, compute, plot_show):
    experiment_name = 'p1v'
    folder_path = os.path.join('.', 'data', 'interim', experiment_name)
    
    # use p1e as inputs
    interim_experiment_name = 'p1e'
    interim_folder_path = os.path.join('.', 'data', 'interim', interim_experiment_name)
    
    for problem_name in problem_list:
        print(problem_name)
        x_train, y_train, x_test, y_test = data_prep.get_bare_data(problem_name=problem_name)
        
        for algo in clustering_algos:
            print(algo)
            optimal_n_clusters = p1v_params[problem_name][algo]
            
            filename = f'{interim_experiment_name}_{problem_name}_{algo}.csv'
            df = io.read_data(interim_folder_path, filename)
            
            fig = plt.figure()
            fig.set_size_inches(fig_dim_x, fig_dim_y)
            # Kmeans
            if algo == 'kmeans':
                clusterer, objective_function, cluster_labels, centers = kmeans.my_kmeans(
                    X=x_train,
                    n_clusters=optimal_n_clusters,
                    random_state=random_state)
                cluster_labels_test = clusterer.predict(x_test)

            # EM GMM
            if algo == 'gmm':
                clusterer, cluster_labels, centers = em_gmm.em_gmm(
                    X=x_train,
                    n_clusters=optimal_n_clusters,
                    random_state=random_state)
                cluster_labels_test = clusterer.predict(x_test)

            # Silhouette
            silhouette_avg, sample_silhouette_values = clustering_evaluation.my_silhouette(
                X=x_train,
                n_clusters=optimal_n_clusters,
                cluster_labels=cluster_labels
                )
                    
            # Dummy Classifier
            bare_train_score, bare_test_score = hf.dummy_classifier(x_train, y_train, x_test, y_test)
            
            x_train_cluster = np.hstack((x_train, hf.one_hot_encode_array(cluster_labels)))
            x_test_cluster  = np.hstack((x_test, hf.one_hot_encode_array(cluster_labels_test, optimal_n_clusters)))
            
            # Save transformed data
            io.save_numpy_array(
                x_train_cluster,
                folder_path,
                filename=f'{experiment_name}_{problem_name}_{algo}_x_train.csv')
            io.save_numpy_array(
                x_test_cluster,
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
            
            
            new_train_score, new_test_score = hf.dummy_classifier(x_train_cluster, y_train, x_test_cluster, y_test)
            
            print(f'{bare_train_score:.2f}', f'{bare_test_score:.2f}')
            print(f'{new_train_score:.2f}', f'{new_test_score:.2f}')
            print()


            # Plotting
            if plot_show:
                
                # Mutual Information
                # mask = (df['problem_name']==problem_name) & (df['clustering_algo']==algo)
                # x = df[mask]['n_clusters']
                # y = df[mask]['mutual_info_train']
                # y2 = df[mask]['mutual_info_test']
                # plt.figure()
                # plt.plot(x, y, label='train')
                # plt.plot(x, y2, label='test')
                # ymin, ymax = plt.gca().get_ylim()
                # plt.vlines(optimal_n_clusters, ymin, ymax, label=f'Max at {optimal_n_clusters}', color='k', linestyles='--')
                # plt.xlabel('Number of Clusters')
                # plt.ylabel('Mutual Information Score')
                # plt.legend()
                # plt.title(f'{algo} used on {problem_name}')
                # filename = f'{experiment_name}_{problem_name}_{algo}_MutualInformation'
                # io.save_fig(experiment_name, filename)
                
                # Fancy silhouette plot
                # clustering_plotting.clustering_plotting(
                #     X=x_train,
                #     x_dim_0=0,
                #     x_dim_1=1,
                #     n_clusters=optimal_n_clusters,
                #     cluster_labels=cluster_labels,
                #     centers=centers,
                #     silhouette_avg=silhouette_avg,
                #     sample_silhouette_values=sample_silhouette_values
                #     )
                
                # # Pair Plot
                # try:
                #     temp_array = np.hstack((x_train, cluster_labels[:,None]))
                #     dummy = list(range(0,12))
                #     to_sns = pd.DataFrame(temp_array)
                #     to_sns.rename(columns={temp_array.shape[1]-1: 'Clusters'}, inplace=True)
                #     palette=sns.color_palette(n_colors=optimal_n_clusters)
                #     pairplot = sns.pairplot(to_sns,
                #                 x_vars=dummy,
                #                 y_vars=dummy,
                #                 hue='Clusters',
                #                 palette=palette,
                #                 )
                #     # plt.legend(title='Clusters', labels=list(range(optimal_n_clusters)))
                #     filename = f'{experiment_name}_{problem_name}_{algo}_PairPlot'
                #     io.save_fig(experiment_name, filename)
                # except KeyError:
                #     print('GOT A PAIR PLOT KEYERROR')
                    

                for label in set(cluster_labels):
                    mask = cluster_labels[:,None] == label
                    mask = mask.squeeze()
                    dim_a = p1v_params[problem_name]['dims'][0]
                    dim_b = p1v_params[problem_name]['dims'][1]
                    x = x_train[:,dim_a][mask]
                    y = x_train[:,dim_b][mask]
                    plt.plot(x,y,'.',label=label)
            plt.xlabel(f'Feature {dim_a}', fontsize=fontsize)
            plt.ylabel(f'Feature {dim_b}', fontsize=fontsize)
            plt.legend(fontsize=fontsize_ticks, title='Clusters', title_fontsize=fontsize)
            plt.xticks(fontsize=fontsize_ticks)
            plt.yticks(fontsize=fontsize_ticks)
            plt.tick_params(direction='in', which='both')
            plt.title(f'{problem_name} using {algo} clustering', fontsize=fontsize_title)
            plt.tight_layout(pad=0)
            # plt.show()
            
            # filename = f'{experiment_name}_{problem_name}_{algo}_{dependent_variable}_vs_{independent_variable}'
            filename = f'part1_validate_{problem_name}_{algo}'
            io.save_fig(experiment_name, filename)
            