import os
import scipy.io as io
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import h5py

path_to_input_data = os.path.join(os.getcwd(), '..', '..', 'data', 
		'NACCS_TS_Sim0_Post0_ST_TROP_STcond.csv')
input_df = pd.read_csv(path_to_input_data, usecols=['Storm ID', 'yyyymmddHHMM'], 
		skiprows=[1,2])

path_to_h5_data = os.path.join('C:\\', 'Users', 'sanderdy', 'Downloads',
		'NACCS_TS_SimB_Post0_SP03945_ADCIRC01_Timeseries.h5')
h5_f = h5py.File(path_to_h5_data, 'r')

path_to_prepocessed_data = os.path.join(os.getcwd(), '..', '..', 'data',
		'DataLengths.csv')
df = pd.read_csv(path_to_prepocessed_data)

df['input_time_hrs'] = df['input_length']
df['output_time_hrs'] = df['mat_len']*(10/60)

input_start = []
input_end =[]
output_start = []
output_end = []
unique_storms = np.unique(df['storm'].values)
for storm in unique_storms:
	temp = input_df.loc[input_df['Storm ID']==storm]
	input_start.append(temp['yyyymmddHHMM'].iloc[0])
	input_end.append(temp['yyyymmddHHMM'].iloc[-1])

	storm_str = 'Synthetic_{0:04d}' .format(storm)
	storm_str = storm_str + ' - {}' .format(storm)
	data = np.array(h5_f[storm_str]['yyyymmddHHMM'])
	output_start.append(data[0])
	output_end.append(data[-1])

df['input_start'] = input_start
df['input_end'] = input_end
df['output_start'] = output_start
df['output_end'] = output_end

df.to_csv('data_lengths.csv', index=False)	
