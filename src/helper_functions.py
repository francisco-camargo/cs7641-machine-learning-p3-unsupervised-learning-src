import numpy as np
import pandas as pd
from matplotlib import colors
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score

def add_confidence_halfband(x, num_samples, z_score=1.96): # 1.96 z-score for 95% confidence
    std_of_mean = x/(num_samples-1)**0.5
    return z_score*std_of_mean

def one_hot_encode_array(arr, num_classes=None):
    if num_classes is None:
        num_classes = arr.max() + 1
    res = np.zeros((arr.size, num_classes))
    res[np.arange(arr.size), arr] = 1
    return res

def reconstruction_error(x_train, learner):
    learner.fit(x_train)
    new_x_train = learner.transform(x_train)
    inverse = learner.inverse_transform(new_x_train)
    return r2_score(x_train, inverse)

def my_f1_score(x_train, y_train, x_test, y_test, clf):
    average='weighted'
    train_score = f1_score(y_train, clf.predict(x_train), average=average)
    test_score = f1_score(y_test, clf.predict(x_test), average=average)
    return train_score, test_score

def dummy_classifier(x_train, y_train, x_test, y_test):
    n_estimators = 10
    learning_rate = 1e0
    random_state=42
    clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)                
    clf.fit(x_train, y_train)
    train_score, test_score = my_f1_score(x_train, y_train, x_test, y_test, clf)
    return train_score, test_score

def calc_kurtosis(X):
    tmp = pd.DataFrame(X)
    avg_kurtosis = tmp.kurt(axis=0).abs().mean()
        # https://github.com/kylewest520/CS-7641---Machine-Learning/blob/master/Assignment%203%20Unsupervised%20Learning/CS%207641%20HW3%20Code.py
    return avg_kurtosis

def threshold_filter(X, threshold):
    X = np.array(X)
    if X[-1]<threshold:
        return len(X), X[-1]
    num_keep_components = np.argmax(X >= threshold) + 1
    cumsum_kept = X[num_keep_components-1]
    return num_keep_components, cumsum_kept

def cumsum_threshold_filter(X, threshold):
    Y = np.cumsum(X)
    num_keep_components, cumsum_kept = threshold_filter(Y, threshold)
    return num_keep_components, cumsum_kept

def pipeline_helper(dict_in, prefix):
    '''
    Helper function to hand-off parameter values to the Pipeline using the proper format
    '''
    return {prefix +'__' + key:val for key, val in dict_in.items()}

# determine transparent color equivalents
# https://stackoverflow.com/questions/33371939/calculate-rgb-equivalent-of-base-colors-with-alpha-of-0-5-over-white-background
def make_rgb_transparent(rgb, bg_rgb=(1,1,1), alpha=0.2):
    return [alpha * c1 + (1 - alpha) * c2 for (c1, c2) in zip(rgb, bg_rgb)]

def get_transparent_color(plot_object, bg_rgb=(1,1,1), alpha=0.2):
    color = plot_object[0].get_color() # get str value of color
    color = colors.colorConverter.to_rgb(color) # convert to tuple value of color
    color = make_rgb_transparent(color, bg_rgb, alpha)
    return color