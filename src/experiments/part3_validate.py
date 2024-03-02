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
from src.config import p3v_params
import src.helper_functions as hf

fontsize = 9
fontsize_ticks = fontsize - 2
fontsize_title = fontsize + 2
fig_dim_x = 3.2
fig_dim_y = fig_dim_x * 0.75
alpha = 0.2

# Step 3 Validation
def run_experiment(problem_list, dim_red_algos, clustering_algos, random_state, compute, plot_show):
    experiment_name = 'p3v'
    interim_experiment_name = 'p3e'
    interim_folder_path = os.path.join('.', 'data', 'interim', interim_experiment_name)
    
    for problem_name in problem_list:
        print(problem_name)
        x_train_original, y_train_original, x_test_original, y_test_original = data_prep.get_bare_data(problem_name=problem_name)
        
        for dim_red_algo in dim_red_algos:
            print(f'\t{dim_red_algo}')
            
            # Get dimensionally reduced datasets (from step 2)
            x_train, y_train, x_test, y_test = data_prep.get_data_post_feature_engineering('p2v', problem_name, algo=dim_red_algo)
            if len(x_train.shape)==1:
                x_train = x_train.reshape(-1, 1)
                x_test  = x_test.reshape(-1, 1)
        
            for clustering_algo in clustering_algos:
                print(f'\t\t{clustering_algo}')
                optimal_n_clusters = p3v_params[problem_name][dim_red_algo][clustering_algo]
                
                # Get unsupervised clustering metrics
                filename = f'{interim_experiment_name}_{problem_name}_{dim_red_algo}_{clustering_algo}.csv'
                df = io.read_data(interim_folder_path, filename)
                
                # Kmeans
                if clustering_algo == 'kmeans':
                    clusterer, objective_function, cluster_labels, centers = kmeans.my_kmeans(
                        X=x_train,
                        n_clusters=optimal_n_clusters,
                        random_state=random_state)
                    cluster_labels_test = clusterer.predict(x_test)

                # EM GMM
                if clustering_algo == 'gmm':
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
                original_train_score, original_bare_test_score = hf.dummy_classifier(x_train_original, y_train_original, x_test_original, y_test_original)
                
                bare_train_score, bare_test_score = hf.dummy_classifier(x_train, y_train, x_test, y_test)
                
                x_train_cluster = np.hstack((x_train, hf.one_hot_encode_array(cluster_labels)))
                x_test_cluster  = np.hstack((x_test, hf.one_hot_encode_array(cluster_labels_test, optimal_n_clusters)))
                new_train_score, new_test_score = hf.dummy_classifier(x_train_cluster, y_train, x_test_cluster, y_test)
                
                print(f'\t\t\t{original_train_score:.2f}', f'{original_bare_test_score:.2f}')
                print(f'\t\t\t{bare_train_score:.2f}', f'{bare_test_score:.2f}')
                print(f'\t\t\t{new_train_score:.2f}', f'{new_test_score:.2f}')
                print()


                # Plotting
                if plot_show:
                    
                    fig = plt.figure()
                    fig.set_size_inches(fig_dim_x, fig_dim_y)
                    
                    for label in set(cluster_labels):
                        mask = cluster_labels[:,None] == label
                        mask = mask.squeeze()
                        try:
                            dim_a = p3v_params[problem_name][dim_red_algo]['dims'][0]
                            dim_b = p3v_params[problem_name][dim_red_algo]['dims'][1]
                        except IndexError:
                            dim_a = 0
                            dim_b = 0
                        x = x_train[:,dim_a][mask]
                        y = x_train[:,dim_b][mask]
                        plt.plot(x,y,'.',label=label)
                    plt.xlabel(f'Feature {dim_a}', fontsize=fontsize)
                    plt.ylabel(f'Feature {dim_b}', fontsize=fontsize)
                    if len(set(cluster_labels)) < 5:
                        plt.legend(fontsize=fontsize_ticks, title='Clusters', title_fontsize=fontsize)
                    plt.xticks(fontsize=fontsize_ticks)
                    plt.yticks(fontsize=fontsize_ticks)
                    plt.tick_params(direction='in', which='both')
                    plt.title(f'{problem_name} {dim_red_algo} {clustering_algo}', fontsize=fontsize_title)
                    plt.tight_layout(pad=0)
                    # plt.show()
                    
                    # filename = f'{experiment_name}_{problem_name}_{algo}_{dependent_variable}_vs_{independent_variable}'
                    filename = f'{experiment_name}_{problem_name}_{dim_red_algo}_{clustering_algo}_scatter'
                    io.save_fig(experiment_name, filename)
                    
                    # # Mutual Information
                    # mask = (df['problem_name']==problem_name) & (df['clustering_algo']==clustering_algo)
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
                    # plt.title(f'{dim_red_algo}_{clustering_algo} used on {problem_name}')
                    # filename = f'{experiment_name}_{problem_name}_{dim_red_algo}_{clustering_algo}_MutualInformation'
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
                    #     dummy = list(range(0, x_train.shape[1]))
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
                    #     filename = f'{experiment_name}_{problem_name}_{dim_red_algo}_{clustering_algo}_PairPlot'
                    #     io.save_fig(experiment_name, filename)
                    # except KeyError:
                    #     print('GOT A PAIR PLOT KEYERROR')
                
                plt.close('all')    