import requests
from neuralNetwork import split_features
import pandas as pd
from showData import date_parser, read_data

URL = "localhost:5000/predict"

data = read_data()