'''
Run KMeans and GMM on two data
'''

import os
import matplotlib.pyplot as plt

import src.data_prep.data_prep as data_prep
import src.data_prep.read_write_data as io
import src.helper_functions as hf
import src.mlp.train as train
import src.mlp.test as test

# Step 4 Validation
def run_experiment(
    mlp_mode,
    problem_list,
    preprocess_algos,
    random_state,
    compute,
    plot_show,
    ):
    
    if preprocess_algos is None:
        preprocess_algos = ['vanilla']
    
    for problem_name in problem_list:
        print()
        print(problem_name)
        if problem_name == 'monks': from src.mlp.mlp_config import monks_mlp_config as train_dict
        if problem_name == 'cancer': from src.mlp.mlp_config import cancer_mlp_config as train_dict
        if problem_name == 'iris': from src.mlp.mlp_config import iris_mlp_config as train_dict
        if problem_name == 'penguins': from src.mlp.mlp_config import penguins_mlp_config as train_dict
        if problem_name == 'wine': from src.mlp.mlp_config import wine_mlp_config as train_dict
        
        for preprocess_algo in preprocess_algos:
            print(f'\t{preprocess_algo}')
            
            # Read in dim reduced data
            if mlp_mode in ['vanilla_e', 'vanilla_v']:
                x_train, y_train, x_test, y_test = data_prep.get_bare_data(problem_name=problem_name)
            else:
                if mlp_mode in ['p4e', 'p4v']: interim_experiment_name = 'p2v'
                elif mlp_mode in ['p5e', 'p5v']: interim_experiment_name = 'p1v'
                x_train, y_train, x_test, y_test = data_prep.get_data_post_feature_engineering(
                    interim_experiment_name,
                    problem_name, algo=preprocess_algo,
                    )
            if len(x_train.shape)==1:
                x_train = x_train.reshape(-1, 1)
                x_test  = x_test.reshape(-1, 1)
            
            print(f'\t\tx_train shape: {x_train.shape}')
                
            # MLP Experiments
            if mlp_mode in ['p4e', 'p5e', 'vanilla_e']:
                # Train MLP
                train.train(
                    preprocess_algo=preprocess_algo,
                    X_TV=x_train,
                    y_TV=y_train,
                    train_dict=train_dict,
                    random_state=random_state,
                    )
                print()
            elif mlp_mode in ['p4v', 'p5v', 'vanilla_v']:
                # Test MLP
                test.test(
                    experiment_name=mlp_mode,
                    problem_name=problem_name,
                    preprocess_algo=preprocess_algo,
                    X_TV=x_train,
                    y_TV=y_train,
                    X_Test=x_test,
                    y_Test=y_test,
                    train_dict=train_dict,
                    random_state=random_state,
                    compute=compute,
                    plot_show=plot_show,
                    )
                plt.close('all')

def plot_fittime(
        experiment_list,
        problem_list,
        algos,
        ):
    
    fontsize = 9
    fontsize_ticks = fontsize - 2
    fig_dim_x = 3.2
    fig_dim_y = fig_dim_x * 0.75
    tight_pad = 0
    lw=2
    
    for problem_name in problem_list:
        fig = plt.figure()
        fig.set_size_inches(fig_dim_x, fig_dim_y)
        for experiment_name in experiment_list:
            folder_path = os.path.join('.', 'data', 'interim', experiment_name)
            for algo in algos:
                filename = f'{experiment_name}_{problem_name}_{algo}.csv'
                try:
                    df = io.read_data(folder_path, filename)
                except FileNotFoundError:
                    continue
                
                x = df['train_sizes']
                y = df['fit_times_mean']
                half_band = df['fit_times_std']
                p = plt.plot(x, y, label=f'{algo}')
                color = hf.get_transparent_color(p)
                plt.fill_between(x,y+half_band,y-half_band, color=color)
                
        plt.tick_params(direction='in')
        plt.xticks(fontsize=fontsize_ticks)
        plt.yscale('log')
        plt.yticks(fontsize=fontsize_ticks)
        plt.ylim([1e-1, 1e1])
        plt.legend(fontsize=fontsize_ticks, ncol=2)
        plt.xlabel('Number of Samples', fontsize=fontsize)
        plt.ylabel(f'{problem_name} Fit Time (s)', fontsize=fontsize)
        fig.tight_layout(pad=tight_pad)
                
        filename = f'{problem_name}_FitTimeCurve'
        io.save_fig('fit_time', filename)
        