import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import tensorflow as tf

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *
from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras import backend as K

from utils import *
from dataApi import *


class DATA:
    def __init__(self):
        # Daten

        data = read_data('filled_split_normalized_data')

        train_df, test_df = train_test_split(data, test_size=0.3)

        train_df_features = train_df.drop(columns=["Absatz"])
        test_df_features = test_df.drop(columns=["Absatz"])

        train_df_target = train_df["Absatz"]
        test_df_target = test_df["Absatz"]

        # plt.show()

        # print("features", train_df_features.shape)
        # print("targets", train_df_target.shape)

        # print("data", data)

        self.train_data = tf.data.Dataset.from_tensor_slices((train_df_features, train_df_target)).shuffle(
            len(train_df_features)).batch(581)
        self.test_data = tf.data.Dataset.from_tensor_slices((test_df_features, test_df_target)).shuffle(
            len(test_df_features)).batch(499)




data = DATA()

# print("Evaluate")
# evaluate_model = load_model("F:\dev\TensorflowNetworks\model\TgBierFill1.h5")
# e_score = evaluate_model.evaluate(data.test_data)
# print("Evaluate Score", e_score)

# Model vars (Hyperparameter)
epochs = 4000
learning_rate = 0.15
optimizer = tf.keras.optimizers.Adam(learning_rate)  # Adam(lr=learning_rate)
batch_size = 499
bias_initializer = Constant(value=0.0)
weight_initializer = RandomUniform(minval=-1.0, maxval=1.0)


model = Sequential()
model.add(Dense(5, activation="relu", bias_initializer=bias_initializer, kernel_initializer=weight_initializer))
model.add(Dense(20, activation="relu", bias_initializer=bias_initializer, kernel_initializer=weight_initializer))
model.add(Dense(20, activation="relu", bias_initializer=bias_initializer, kernel_initializer=weight_initializer))
model.add(Dense(1))

save_dir = os.path.abspath("C:/Users/vikto/PycharmProjects/TgBier/models/")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
model_path = os.path.join(save_dir, "TgBier1.h5")

# Log Path
path = get_tb_dir() + "TgBier/f2"
log_dir = os.path.abspath(path)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

model.compile(loss="mae",
              optimizer=optimizer,
              metrics=['mae'])

tb_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True
)

plateau_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.90,  # Reduktion der LR um 5%
    patience=20,  # 20
    verbose=1,
    min_lr=0.0000001  # 0.0001
)

history = model.fit(
    data.train_data,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=data.test_data,
    callbacks=[
        tb_callback,
        plateau_callback])

score = model.evaluate(data.test_data)
print("Calc Score: ", score)

model.save(model_path)
