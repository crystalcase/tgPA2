import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
from utils import *
from sklearn import preprocessing
from dataApi import *

data = read_data('filled_split_data')
min, max = get_min_max(data)
print('min', min)
print('max', max)

scaler = preprocessing.MinMaxScaler()
scaler.fit_transform(np.array(data['Absatz']).reshape(-1, 1))

n4 = scaler.inverse_transform(np.array([[0.1686]]))
n5 = scaler.inverse_transform(np.array([[0.18918918]]))

print("n4", n4)
print("n5", n5)

# print("data", data)

x = data['Kalendertag']
y = data['Absatz']
"""""
ax = plt.gca()
formatter = md.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(formatter)
plt.plot(x, y, label="Label")
plt.xlabel('Zeit')
plt.ylabel('SumAbsatz')
plt.title('All Data')
plt.legend()
#plt.show()
"""