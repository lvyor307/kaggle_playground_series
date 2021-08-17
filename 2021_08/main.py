import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

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
df_train: pd.DataFrame = pd.read_csv('train.csv')  # read data

# Split to features (X) and lable (y)
data_train: np.ndarray = df_train.values
X_train: np.ndarray = data_train[:, 1:-1]
y_train: np.ndarray = data_train[:, -1]


"""
Random Forest Model
"""

# Number of features to consider at every split
max_features = [int(x) for x in np.linspace(start=1, stop=20, num=2)]
# Minimum number of samples required to split a node
min_samples_split = [5, 10]
# Minimum number of samples required at each leaf node


# Create the grid
grid = {'max_features': max_features,
        'min_samples_split': min_samples_split,
        }

# define model
rf = RandomForestRegressor()
# define search
grid_search = GridSearchCV(estimator=rf,
                            param_grid=grid,
                            scoring='neg_root_mean_squared_error',
                            cv=3,
                            verbose=2,
                            n_jobs=-1)
# perform the search
results = grid_search.fit(X_train, y_train)


"""
 Get the paramters that gives the best RMSE
"""

best_grid = grid_search.best_estimator_

rf_cv = RandomForestRegressor(max_features=best_grid.get('max_features'),
                              min_samples_split=best_grid.get('min_samples_split')
                              )

rf_cv.fit(X_train, y_train)


"""
Predictions to test set
"""
df_test: pd.DataFrame = pd.read_csv('test.csv')  # read data

data_test: np.ndarray = df_test.values
X_test: np.ndarray = data_test[:, 1:-1]

loss = rf_cv.predict(X_test)

prediction: pd.DataFrame = pd.DataFrame({
    'id': df_test.values[:, 0].astype(int),
    'loss': loss
})

prediction.to_csv('prediction.csv', index=False)
