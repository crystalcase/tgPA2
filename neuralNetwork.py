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


def split_features(data_file):

    df = pd.DataFrame()
    for date in data_file["Kalendertag"]:
        new_row = pd.DataFrame(
            data={
                "Kalendertag": [date.day],
                "Monat": [date.month],
                "Jahr": [date.year],
                "Kalenderwoche": [date.weekofyear],
                "Wochentag": [date.dayofweek],
            },
        )
        df = df.append(new_row)

    df2 = []
    for sell in data_file["Absatz"]:
        df2.append(sell)
    df["Absatz"] = normalize_sells(df2)
    return df


def normalize_sells(sell_values):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(np.array(sell_values).reshape(-1, 1))


def save_data_callback(y_true, y_pred):
    global output_data, i
    try:
        output_data
    except NameError:
        output_data = None

    try:
        i
    except NameError:
        i = None

    new_row = pd.DataFrame(
        data={
            'y_true': [y_true],
            'y_pred': [y_pred]
        }
    )

    if output_data is None:
        output_data = new_row
    else:
        output_data.append(new_row)

    if i is None:
        i = 0
    else:
        i += 1
    print("Counter i", i)
    print("output_data", output_data)
    return 0.0


class DATA:
    def __init__(self):
        # Daten
        self.data_file = read_data()

        self.data_file['Absatz'] = self.data_file['Absatz'].astype(np.float32)

        data_df = split_features(self.data_file)

        self.raw_data = self.data_file

        train_df, test_df = train_test_split(data_df, test_size=0.3)

        train_df_features = train_df.drop(columns=["Absatz"])
        test_df_features = test_df.drop(columns=["Absatz"])

        train_df_target = train_df["Absatz"]
        test_df_target = test_df["Absatz"]

        # sns.pairplot(data_df[["Kalendertag", "Monat", "Jahr", "Kalenderwoche", "Wochentag", "Absatz"]], diag_kind="kde")
        # plt.show()

        print("features", train_df_features.shape)
        print("targets", train_df_target.shape)

        self.train_data = tf.data.Dataset.from_tensor_slices((train_df_features, train_df_target)).shuffle(
            len(train_df_features)).batch(293)
        self.test_data = tf.data.Dataset.from_tensor_slices((test_df_features, test_df_target)).shuffle(
            len(test_df_features)).batch(181)

    def get_raw_data(self):
        return self.data_file


data = DATA()
# Model vars (Hyperparameter)
epochs = 1300
learning_rate = 0.1
optimizer = tf.keras.optimizers.RMSprop(0.001)  # Adam(lr=learning_rate)
batch_size = 2231
bias_initializer = Constant(value=0.0)
weight_initializer = RandomUniform(minval=-1.0, maxval=1.0)


class SequentialWithSave(keras.Sequential):
    def train_step(self, original_data):
        data = data_adapter.expand_1d(original_data)
        x, y_true, w = data_adapter.unpack_x_y_sample_weight(data)
        y_pred = self(x, training=True)

        K.print_tensor(x, "Batch output (x) = ")
        K.print_tensor(y_true, "Batch output (y_true) = ")
        K.print_tensor(y_pred, "Batch output (y_pred) = ")

        result = super().train_step(original_data)

        return result


model = Sequential()
model.add(Dense(5, activation="relu", bias_initializer=bias_initializer, kernel_initializer=weight_initializer))
model.add(Dense(10, activation="relu", bias_initializer=bias_initializer, kernel_initializer=weight_initializer))
model.add(Dense(1))

save_dir = os.path.abspath("F:/dev/TensorflowNetworks/model/")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
model_path = os.path.join(save_dir, "TgBierNew.h5")

# Log Path
log_dir = os.path.abspath("C:/dev/TensorflowNetworks/logs/TgBier/new")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae', 'mse'])

tb_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True
)

plateau_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.95,  # Reduktion der LR um 5%
    patience=5,  # 20
    verbose=1,
    min_lr=0.0000001  # 0.0001
)

history = model.fit(
    data.train_data,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=data.test_data,
    callbacks=[tb_callback, plateau_callback])

score = model.evaluate(data.test_data)
print("Score: ", score)

# model.save(model_path)

