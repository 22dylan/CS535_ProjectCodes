import os
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')


storm_idx = 632
sp_idx = 1000
path_to_data = os.path.join(os.getcwd(), '..', '..', 'data', 'CHS_Storms_raw')
file = os.path.join(path_to_data, os.listdir(path_to_data)[storm_idx-1])
data = io.loadmat(file)['surge']
print(file)
dt=10
time = np.linspace(0, len(data)*dt, len(data))
print(np.shape(data))
plt.figure()
plt.plot(time/60, data[:,sp_idx])
plt.title("Storm 0468\nSave Point {}" .format(sp_idx+1))
plt.xlabel('Time (hrs.)')
plt.ylabel('Height (m.)')

plt.savefig('temp_surge.png', transparent=True)
plt.show()
