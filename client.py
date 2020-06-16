import requests
from neuralNetwork import split_features
import pandas as pd
from utils import *
from dataApi import *

URL = "localhost:5000/predict"

data = read_data('split_normalized_data')

# for index, data in data.iterrows():


#TODO Client