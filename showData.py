import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
from utils import *


data = get_summed_data()
min, max = get_min_max(data)
print('min', min)
print('max', max)


data['Kalendertag'].replace('', np.nan, inplace=True)
data.dropna(subset=['Kalendertag'], inplace=True)

x = data['Kalendertag']
y = data['SumAbsatz']

ax = plt.gca()
formatter = md.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(formatter)
plt.plot(x, y, label="Label")
plt.xlabel('Zeit')
plt.ylabel('SumAbsatz')
plt.title('All Data')
plt.legend()
plt.show()
