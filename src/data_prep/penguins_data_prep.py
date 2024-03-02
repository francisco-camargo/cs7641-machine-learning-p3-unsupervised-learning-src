'''
Script to clean up penguins dataset
    # https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data
'''

import os
import pandas as pd
from sklearn.model_selection import train_test_split

import read_write_data as io

def add_one_hot(df, column_name):
    # Get one hot encoding of columns B
    one_hot = pd.get_dummies(df[column_name], prefix=column_name)
    # Drop column B as it is now encoded
    df = df.drop(column_name, axis = 1)
    # Join the encoded df
    return df.join(one_hot)

# Import raw Penguins data
folder_path = os.path.join('.', 'data', 'raw')
filename = 'penguins_lter.csv'
df = io.read_data(folder_path, filename)

# Define which columns to keep; target and features
keep_columns = [
    'Species',
    # 'Island',
    'Culmen Length (mm)',
    'Culmen Depth (mm)',
    'Flipper Length (mm)',
    'Body Mass (g)',
]
df = df[keep_columns]

# Drop rows
df.dropna(inplace=True)

# Define which columns get one-hot encoding
# one_hot_col = [
#     'Island',
# ]
# for col in one_hot_col:
#     df = add_one_hot(df, col)
    
# Rename Species names
rename_dict = {
    'Chinstrap penguin (Pygoscelis antarctica)': 'Chinstrap',
    'Adelie Penguin (Pygoscelis adeliae)': 'Adelie',
    'Gentoo penguin (Pygoscelis papua)': 'Gentoo',
}
df['Species'] = [rename_dict[row_value] for row_value in df['Species']]

# split into test and train
test_size = 0.2
df_train, df_test = train_test_split(df, test_size=test_size, random_state=42, stratify=df['Species'])

# Save as pickle files
path_train = os.path.join('.', 'data', 'interim', 'penguins_train.pkl')
df_train.to_pickle(path_train)
path_test  = os.path.join('.', 'data', 'interim', 'penguins_test.pkl')
df_test.to_pickle(path_test)
