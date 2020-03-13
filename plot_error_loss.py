#Plots results from csv files

import os
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn')

#Select which key to plot results for
BATCH_SIZE = 50     # mini_batch size
MAX_EPOCH = 100      # maximum epoch to train
hidden_size = 25    # size of hidden layer
n_layers = 1        # number of lstm layers

key = 'B{}_h{}_l{}' .format(BATCH_SIZE, hidden_size, n_layers)


path_to_results = os.path.join(os.getcwd(), 'LSTM_training_results', '{}_results.csv'.format(key))
results = pd.read_csv(path_to_results)
# os.path.join(path_to_results, os.listdir(path_to_data)[storm_idx-1])

#plot acc
plt.figure(1)
plt.plot(results['train_acc'])
plt.plot(results['test_acc'])
plt.legend(['train_acc','test_acc'])
plt.title('Average Error')
plt.xlabel('Epoch #')
plt.ylabel('Error (m)')
#save error plot
plt.savefig('result_plots/{}_Error_plot.png'.format(key), transparent=False)

#plot loss
plt.figure(2)
plt.plot(results['train_loss'])
plt.plot(results['test_loss'])
plt.legend(['train_loss','test_loss'])
plt.title('Loss')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
#save loss plot
plt.savefig('result_plots/{}_Loss_plot.png'.format(key), transparent=False)


plt.show()