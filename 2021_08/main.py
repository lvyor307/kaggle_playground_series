import os
import pandas as pd
import numpy as np


# Get the data with kaggle API
filename: str = 'tabular-playground-series-aug-2021'

if not os.path.isfile(filename + '.zip'):
    os.system("kaggle competitions download -c " + filename)

if not os.path.isfile('test.csv') and os.path.isfile(filename + '.zip'):
    os.system('unzip ' + filename + '.zip')
    os.system('rm tabular-playground-series-aug-2021.zip')

if os.path.isfile('test.csv') and os.path.isfile(filename + '.zip'):
    os.system('rm tabular-playground-series-aug-2021.zip')
