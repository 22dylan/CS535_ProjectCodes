import os
import pandas as pd
import numpy as np
import scipy.io as io

from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

class CHS_DataSet(Dataset):

	def __init__(self, path_to_data, xmin, xmax, ymin, ymax, ts_input=False,
		ts_output=False):

		self.ts_output = ts_output

		self.savepoints = self.identify_savepoints(path_to_data, 
										xmin, xmax, ymin, ymax)
		# getting input data

		self.storm_conds = self.read_storm_conds(path_to_data, ts_input)
		if self.ts_output == False:
			self.target = self.read_data_max(path_to_data, self.savepoints)

	def identify_savepoints(self, path_to_data, xmin, xmax, ymin, ymax):
		path_to_data = os.path.join(path_to_data, 
				'NACCS_SavePts_18977_ConversionKey_MSL_to_NAVD_meters.csv')
		data = pd.read_csv(path_to_data, skiprows=[1])

		# trimming points between bounding box
		data = data[data['SP_Longitude'].between(xmin, xmax, inclusive=True)]
		data = data[data['SP_Latitude'].between(ymin, ymax, inclusive=True)]
		return data['SavePointID']

	def read_storm_conds(self, path_to_data, ts_input):
		""" reading the input storm conditions. these are used as input to the 
			neural net"""
		missing_storms = os.path.join(path_to_data, 'MissingStorms_20.txt')
		missing_storms = pd.read_csv(missing_storms, sep=" ", header=None)
		if ts_input == True:
			path_to_data = os.path.join(path_to_data, 
						'NACCS_TS_Sim0_Post0_ST_TROP_STcond.csv')
			df = pd.read_csv(
					path_to_data, 
					skiprows = [1,2],
					usecols=['Storm ID', 'Central Pressure', 
						'Far Field Pressure', 'Forward Speed', 'Heading', 
						'Holland B1', 'Radius Max Winds', 'Radius Pressure 1', 
						'Storm Latitude', 'Storm Longitude']
					)

			df = df[~df['Storm ID'].isin(missing_storms[0])]
			data_temp = df.values
			unique_storms = np.unique(data_temp[:,0])
			data = np.empty((len(unique_storms), 337, 9))
			data[:] = np.nan
			for i, storm_id in enumerate(unique_storms):
				storm_data = data_temp[data_temp[:,0]==storm_id, 1:]
				pad_width = 337 - len(storm_data) 
				storm_data = np.pad(storm_data, ((0,pad_width), (0,0)), 
							'constant', constant_values=np.nan)
				data[i] = storm_data

		elif ts_input == False:
			path_to_data = os.path.join(path_to_data, 'StormConditions.csv')
			df = pd.read_csv(path_to_data)
			df = df[~df['StormID'].isin(missing_storms[0])]
			data = df[['StormID', 'TrackID', 'CentralPressureDeficit', 
						'RadiusMaxWinds','TranslationalSpeed']].values
	
		return data

	def read_data_max(self, path_to_data, savepoints):
		""" Reading the output/target conditions.
			Returns a matrix of maximum surge vlaues. 
			Each row corresponds to a storm, and each
			column corresponds to a save point."""
		path_to_data = os.path.join(path_to_data, 'Max_Surge', 'max_surge.csv')
		df = pd.read_csv(path_to_data)
		sp_list = ['Storm_ID']
		for sp in savepoints:
			sp_list.append('sp_{}'.format(sp))

		df = df[sp_list]
		df.set_index(['Storm_ID'], inplace=True)
		data = df.values
		return data

	def read_data_ts(self, path_to_data, savepoints, storm):
		path_to_data = os.path.join(path_to_data, 'CHS_Storms_raw')
		file_name = ('NACCS_TP_{0:04d}_SYN_Tides_0_SLC_0_RFC_0_surge_all.mat' 
							.format(storm))
		mat_file = os.path.join(path_to_data, file_name)
		data = io.loadmat(mat_file)['surge']	# reading in data
		data = data[:,savepoints-1]		# isolating savepoints in dataset

		""" padding data with nan values so that all time series are the same 
			size. the value of 1980 is used b/c it's the largest of all time
			series """
		pad_width = 1980 - np.shape(data)[0] 
		data = np.pad(data, ((0,pad_width), (0,0)), 
					'constant', constant_values=np.nan)
		return data

	def __len__(self):
		return len(self.storm_conds)

	def __getitem__(self, idx):
		storms = self.storm_conds[:,0]
		data_val = self.storm_conds[:,1:]
		data_val = data_val[idx]

		if self.ts_output == False:
			target = self.target[idx]
		else:
			target = self.read_data_ts(path_to_data, self.savepoints, storms[idx])

		return data_val, target

if __name__ == "__main__":
	# path to data
	path_to_data = os.path.join(os.getcwd(), '..', 'data')
	
	""" defining bounding box """
	# # large bounding box
	# xmin, xmax = -74.619, -73.397
	# ymin, ymax = 40.080, 40.892

	# smaller bounding box
	xmin, xmax = -74.2754, -73.9374
	ymin, ymax = 40.4041, 40.6097
	
	train_test_split = 0.8		# ratio to split test and train data

	# dataset class
	dataset = CHS_DataSet(path_to_data, xmin, xmax, ymin, ymax, ts_input=True,
		ts_output=False)
	print('setup dataset class')

	# computing size of train and test datasets
	train_size = int(train_test_split * len(dataset))
	test_size = len(dataset) - train_size
	lengths = [train_size, test_size]

	# splitting the data into train and test sets
	trn_ds, tst_ds = random_split(dataset, lengths)
	print('split data into train and test sets')

	# setting up train and test dataloaders
	trn_loader = DataLoader(trn_ds, batch_size=50, shuffle=True, num_workers=0)
	tst_loader = DataLoader(tst_ds, batch_size=50, shuffle=True, num_workers=0)
	
	print('converted datasets to dataloaders')
	for i, data in enumerate(trn_loader, 0):
		inputs, targets = data

		# neural network



