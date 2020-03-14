#Plots error distribution from csv files

import os
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches

plt.style.use('seaborn')

#Select which key to plot results for
BATCH_SIZE = 50     # mini_batch size
MAX_EPOCH = 100      # maximum epoch to train
hidden_size = 100    # size of hidden layer
n_layers = 1        # number of lstm layers
box_size = 'S' #'M' 'L'

key = 'LSTM_B{}_h{}_l{}_bb{}' .format(BATCH_SIZE, hidden_size, n_layers, box_size)


path_to_predict = os.path.join(os.getcwd(), 'LSTM_training_results', 'Model_results', '{}_predict.csv'.format(key))
predict = pd.read_csv(path_to_predict)
predict = predict.values


path_to_target = os.path.join(os.getcwd(), 'LSTM_training_results', 'Model_results', '{}_target.csv'.format(key))
target = pd.read_csv(path_to_target)
target = target.values

error = abs(target-predict)
error_avg = error.mean(axis=1)
error = error.flatten()

blue_patch = mpatches.Patch(color='blue', label='Errors')
green_patch = mpatches.Patch(color='green', label='Errors of each savepoint averaged over storms')


plt.figure(1)
sns.kdeplot(error)
sns.kdeplot(error_avg)
plt.legend(handles=[blue_patch, green_patch])
plt.xlabel('error (m)')
plt.ylabel('distribution density')
plt.title('Error Range Distribution')
plt.savefig('LSTM_training_results/result_plots/{}_Error_dist.png'.format(key), transparent=False)
plt.show()
