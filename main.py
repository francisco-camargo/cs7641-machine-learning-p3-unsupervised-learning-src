import src.experiments.part1_explore as part1_explore
import src.experiments.part1_validate as part1_validate
import src.experiments.part2_explore as part2_explore
import src.experiments.part2_validate as part2_validate
import src.experiments.part3_explore as part3_explore
import src.experiments.part3_validate as part3_validate
import src.experiments.part4 as part4

random_state=42

compute     = True
plot_show   = True

experiment_list = [
    'p1e',
    # 'p1v',
    # 'p2e',
    # 'p2v',
    # 'p3e',
    # 'p3v',
    # 'vanilla_e', # generate MLP models without either clustering or dim reduction
    # 'p5e',
    # 'p4e',
    # 'vanilla_v',
    # 'p5v',
    # 'p4v',
    ]

problem_list = [
    'penguins',
    'wine',
    ]

clustering_algos = [
    'kmeans',
    'gmm',
    ]

dim_red_algos = [
    'pca',
    'ica',
    'rca',
    'rfe',
    ]

# Step 1 Clustering Explore:
if 'p1e' in experiment_list:
    part1_explore.run_experiment(
        problem_list,
        clustering_algos,
        random_state,
        compute,
        plot_show,
        )
        
# Step 1 Clustering Validation:
if 'p1v' in experiment_list:
    part1_validate.run_experiment(
        problem_list,
        clustering_algos,
        random_state,
        compute,
        plot_show,
        )

# Step 2: Dimensionality Reduction Explore
if 'p2e' in experiment_list:
    if compute:
        part2_explore.run_experiment(
            problem_list,
            dim_red_algos,
            random_state,
            )
    if plot_show:
        part2_explore.plot_experiment(
            problem_list,
            dim_red_algos,
            )

# Step 2: Dimensionality Reduction Validation
if 'p2v' in experiment_list:
    part2_validate.run_experiment(
        problem_list,
        dim_red_algos,
        random_state,
        compute,
        plot_show,
        )

# Step 3: (Dimensionality Reduction -> Clustering) Explore
if 'p3e' in experiment_list:
    if compute:
        part3_explore.run_experiment(
            problem_list,
            dim_red_algos,
            clustering_algos,
            random_state,
            compute,
            plot_show,
            )
    if plot_show:
        part3_explore.plot_experiment(
            problem_list,
            dim_red_algos,
            clustering_algos,
            )

# Step 3: (Dimensionality Reduction -> Clustering) Validate
if 'p3v' in experiment_list:
    part3_validate.run_experiment(
        problem_list,
        dim_red_algos,
        clustering_algos,
        random_state,
        compute,
        plot_show,
        )
    
# Vanilla
if 'vanilla_e' in experiment_list:
    part4.run_experiment(
        'vanilla_e',
        problem_list,
        None,
        random_state,
        compute,
        plot_show,
        )
if compute and ('vanilla_v' in experiment_list):
    part4.run_experiment(
        'vanilla_v',
        problem_list,
        None,
        random_state,
        compute,
        plot_show,
        )
    
# Step 5: (Clustering -> MLP) Explore
if 'p5e' in experiment_list:
    part4.run_experiment(
        'p5e',
        problem_list,
        clustering_algos,
        random_state,
        compute,
        plot_show,
        )
    
# Step 5: (Clustering -> MLP) Validation
if compute and ('p5v' in experiment_list):
    part4.run_experiment(
        'p5v',
        problem_list,
        clustering_algos,
        random_state,
        compute,
        plot_show,
        )
    
# Step 4: (Dimensionality Reduction -> MLP) Explore
if 'p4e' in experiment_list:
    part4.run_experiment(
        'p4e',
        problem_list,
        dim_red_algos,
        random_state,
        compute,
        plot_show,
        )
    
# Step 4: (Dimensionality Reduction -> MLP) Validation
if compute and ('p4v' in experiment_list):
    part4.run_experiment(
        'p4v',
        problem_list,
        dim_red_algos,
        random_state,
        compute,
        plot_show,
        )
    
# Plot fit times
if plot_show and (('p4v' in experiment_list) or ('p5v' in experiment_list) or ('vanilla_v' in experiment_list)):
    if 'vanilla_v' in experiment_list: algos = ['vanilla']
    else: algos = []
    algos += dim_red_algos + clustering_algos
    part4.plot_fittime(
        experiment_list,
        problem_list,
        algos,
        )
    