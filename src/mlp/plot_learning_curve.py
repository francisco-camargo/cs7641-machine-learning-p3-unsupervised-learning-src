"""
========================
Plotting Learning Curves
========================

Using the following for reference:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

"""

import matplotlib.pyplot as plt

import src.helper_functions as hf

def plot_learning_curve(
    df,
    problem_name,
    preprocess_algo,
    ):
    
    # Plot Learning Curve
    
    fontsize = 9
    fontsize_ticks = fontsize - 2
    fontsize_title = fontsize + 2
    fig_dim_x = 3.2
    fig_dim_y = fig_dim_x * 0.75
    lw=2
    tight_pad = 0
    # xlim = [0,350]
    
    if preprocess_algo != 'vanilla':
        title = f'{problem_name} {preprocess_algo} + MLP Learning Curve'
    else:
        title = f'{problem_name} MLP Learning Curve'
    
    fig = plt.figure()
    fig.set_size_inches(fig_dim_x, fig_dim_y)
    
    x = df['train_sizes']
    y = df['train_scores_mean']
    half_band = df['train_scores_std']
    
    p_train = plt.plot(x, y, color="r", label="Training",lw=lw)
    color = hf.get_transparent_color(p_train)
    plt.fill_between(x,y+half_band,y-half_band, color=color)
    
    y = df['validation_scores_mean']
    half_band = df['validation_scores_std']
    p_validation = plt.plot(x, y, color="g", label="Validation",lw=lw)
    color = hf.get_transparent_color(p_validation)
    plt.fill_between(x,y+half_band,y-half_band, color=color)
    
    plt.title(title, fontdict = {'fontsize' : fontsize_ticks})
    plt.xlabel('Number of Samples', fontsize=fontsize)
    plt.ylabel('f1-weighted score', fontsize=fontsize)
    plt.legend(loc='lower right', fontsize=fontsize_ticks)
    plt.tick_params(direction='in', which='both')
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    # plt.xlim(xlim)
    plt.ylim([0.5,1])
    fig.tight_layout(pad=tight_pad)

    # Save figures
    if preprocess_algo in ['pca', 'ica', 'rca', 'rfe']: experiment_name = 'p4v'
    if preprocess_algo in ['kmeans', 'gmm']: experiment_name = 'p5v'
    if preprocess_algo == 'vanilla': experiment_name = 'vanilla'

    filename = f'{experiment_name}_{problem_name}_{preprocess_algo}_LearningCurve'
    import src.data_prep.read_write_data as io
    io.save_fig(experiment_name, filename)
    