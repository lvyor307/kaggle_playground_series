import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

"""
Get the data with kaggle API
"""

filename: str = 'tabular-playground-series-aug-2021'

if not os.path.isfile(filename + '.zip') and not os.path.isfile('test.csv'):
    os.system("kaggle competitions download -c " + filename)

if not os.path.isfile('test.csv') and os.path.isfile(filename + '.zip'):
    os.system('unzip ' + filename + '.zip')
    os.system('rm tabular-playground-series-aug-2021.zip')

if os.path.isfile('test.csv') and os.path.isfile(filename + '.zip'):
    os.system('rm tabular-playground-series-aug-2021.zip')

""""""

"""
Read data and split to train test
"""

df: pd.DataFrame = pd.read_csv('train.csv')  # read data

# Split to features (X) and lable (y)
data: np.ndarray = df.values
X: np.ndarray = data[:, 1:-1]
y: np.ndarray = data[:, -1]

# Split data to train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=217)

""""""

"""
Random Forest Model
"""
# Number of features to consider at every split
max_features = [int(x) for x in np.linspace(start=1, stop=99, num=2)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'max_features': max_features,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# define model
rf = RandomForestRegressor()
# define model evaluation method
cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=217)
# define search
search = GridSearchCV(estimator=rf,
                      param_grid=random_grid,
                      scoring='neg_root_mean_squared_error',
                      cv=cv,
                      n_jobs=-1)
# perform the search
results = search.fit(X_train, y_train)

a = 1
print('a')
