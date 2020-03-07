import os
import pandas as pd
import numpy as np
import scipy.io as io

from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

class CHS_DataSet(Dataset):

	def __init__(self, path_to_data, xmin, xmax, ymin, ymax, max_surge=True):
		self.max_surge = max_surge
		self.savepoints = self.identify_savepoints(path_to_data, 
										xmin, xmax, ymin, ymax)
		# getting input data
		self.storm_conds = self.read_storm_conds(path_to_data)
		if self.max_surge == True:
			self.target = self.read_datamax(path_to_data, self.savepoints)

	def identify_savepoints(self, path_to_data, xmin, xmax, ymin, ymax):
		path_to_data = os.path.join(path_to_data, 
				'NACCS_SavePts_18977_ConversionKey_MSL_to_NAVD_meters.csv')
		data = pd.read_csv(path_to_data, skiprows=[1])

		# trimming points between bounding box
		data = data[data['SP_Longitude'].between(xmin, xmax, inclusive=True)]
		data = data[data['SP_Latitude'].between(ymin, ymax, inclusive=True)]
		return data['SavePointID']

	def read_storm_conds(self, path_to_data):
		""" reading the input storm conditions. these are used as input to the 
			neural net"""
		missing_storms = os.path.join(path_to_data, 'MissingStorms_20.txt')
		missing_storms = pd.read_csv(missing_storms, sep=" ", header=None)
		path_to_data = os.path.join(path_to_data, 'StormConditions.csv')
		df = pd.read_csv(path_to_data)
		df = df[~df['StormID'].isin(missing_storms[0])]
		data = df[['StormID', 'TrackID', 'CentralPressureDeficit', 
					'RadiusMaxWinds','TranslationalSpeed']].values
		return data

	def read_datamax(self, path_to_data, savepoints):
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

	def read_data(self, path_to_data, savepoints, storm):
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
		if self.max_surge == True:
			target = self.target[idx]
		else:
			target = self.read_data(path_to_data, self.savepoints, storms[idx])

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
	dataset = CHS_DataSet(path_to_data, xmin, xmax, ymin, ymax, max_surge=False)
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



