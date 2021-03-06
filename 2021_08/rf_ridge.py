import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



"""
Get the data with kaggle API
"""
def get_data_from_kaggle_with_API(filename: str):
    """
    this function checks if you have downloaded the data to the current path.
    If yes it checks if the data is unzipped and the 'test.csv' file exists.
    If no it unzips the downloaded zip data and then delete the zipped file from the current path 
    """
    if not os.path.isfile(filename + '.zip') and not os.path.isfile('test.csv'):
        os.system("kaggle competitions download -c " + filename)

    if not os.path.isfile('test.csv') and os.path.isfile(filename + '.zip'):
        os.system('unzip ' + filename + '.zip')
        os.system('rm tabular-playground-series-aug-2021.zip')

    if os.path.isfile('test.csv') and os.path.isfile(filename + '.zip'):
        os.system('rm tabular-playground-series-aug-2021.zip')


"""
Read data and split to train test
"""
def read_data_and_split_to_train_test():
    df_train: pd.DataFrame = pd.read_csv('train.csv')  # read data

    # Split to features (X) and lable (y)
    data_train: np.ndarray = df_train.values
    X: np.ndarray = data_train[:, 1:-1]
    y: np.ndarray = data_train[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=217)
    return X_train, X_test, y_train, y_test


"""
Random Forest Model
"""
def applay_random_forest_to_data(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    # Number of features to consider at every split
    max_features = [int(x) for x in np.linspace(start=3, stop=99, num=10)]
    # Minimum number of samples required to split a node
    min_samples_split = [5, 10]
    # Create the grid
    grid = {'max_features': max_features,
            'min_samples_split': min_samples_split,
            }

    # define model
    rf = RandomForestRegressor()
    # define search
    grid_search_rf = GridSearchCV(estimator=rf,
                                param_grid=grid,
                                scoring='neg_root_mean_squared_error',
                                cv=3,
                                verbose=2,
                                n_jobs=-1)
    # perform the search
    results = grid_search_rf.fit(X_train, y_train)


    best_model_rf = grid_search_rf.best_estimator_
    best_model_rf.fit(X_train, y_train)

    loss = best_model_rf.predict(X_test)
    return loss, best_model_rf


"""
Ridge regression model
"""
def applay_ridge_reg_to_data(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    ridge = Ridge()
    grid = dict()
    grid['alpha'] = [x for x in np.linspace(start=0, stop=4, num=100)]

    grid_search_ridge = GridSearchCV(estimator=ridge,
                                    param_grid=grid,
                                    scoring='neg_root_mean_squared_error',
                                    cv=3,
                                    verbose=2,
                                    n_jobs=-1)

    results = grid_search_ridge.fit(X_train, y_train)
    best_model_ridge = grid_search_ridge.best_estimator_
    best_model_ridge.fit(X_train, y_train)

    loss = best_model_ridge.predict(X_test)
    return loss, best_model_ridge


def models_evaluation(loss_rf,loss_ridge, y_test):
    MSE_rf =  mean_squared_error(loss_rf,y_test)
    R2_rf = r2_score(loss_rf,y_test)

    MSE_ridge = mean_squared_error(loss_ridge,y_test)
    R2_ridge = r2_score(loss_ridge,y_test)
    return MSE_rf, MSE_ridge


def write_presictions_to_csv(best_model):
    df_test: pd.DataFrame = pd.read_csv('test.csv')  # read data

    data_test: np.ndarray = df_test.values
    X_test: np.ndarray = data_test[:, 1:]  

    loss = best_model.predict(X_test)
    prediction: pd.DataFrame = pd.DataFrame({
        'id': df_test.values[:, 0].astype(int),
        'loss': loss
    })
    prediction.to_csv('prediction.csv', index=False)

if __name__ == "__main__":
    filename: str = 'tabular-playground-series-aug-2021'
    get_data_from_kaggle_with_API(filename)
    X_train, X_test, y_train, y_test = read_data_and_split_to_train_test()
    loss_rf, model_rf = applay_random_forest_to_data(X_train, X_test, X_test)
    loss_ridge, model_ridge = applay_ridge_reg_to_data(X_train, y_train, X_test)
    MSE_rf, MSE_ridge = models_evaluation(loss_rf,loss_ridge, y_test)
    if MSE_rf < MSE_ridge:
        write_presictions_to_csv(model_rf)
    else:
        write_presictions_to_csv(model_ridge)