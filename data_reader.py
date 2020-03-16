import os
import pandas as pd
import numpy as np
import scipy.io as io

from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize

class CHS_DataSet(Dataset):

	def __init__(self, path_to_data, xmin, xmax, ymin, ymax, ts_input=False,
					ts_output=False, pad_type=np.nan, ts_delete_step_size=0):
		self.ts_output = ts_output
		self.pad_type = pad_type
		self.savepoints = self.identify_savepoints(path_to_data, 
										xmin, xmax, ymin, ymax)

		# getting input data
		self.storm_conds = self.read_storm_conds(path_to_data, ts_input, 
													ts_delete_step_size)
		""" if not using time series, read the entire output data set while
			initiliazing the class. """
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

	def read_storm_conds(self, path_to_data, ts_input, ts_delete_step_size):
		""" reading the input storm conditions. these are used as input to the 
			neural net"""
		missing_storms = os.path.join(path_to_data, 'MissingStorms_20.txt')
		missing_storms = pd.read_csv(missing_storms, sep=" ", header=None)
		if ts_input == True:
			""" - if reading in the time series as input, each storm has an input
					with dimentions of 337x9. 
				- 337 is used b/c it's the longest of the input time series
				- all others are padded with NaNs
				- input columsn are as follows:
					0: Central Pressure
					1: Far Field pressure
					2: Forward Speed
					3: Heading
					4: Holland B1
					5: Radius Max Winds
					6: Radius pressure diff
					7: Latitude
					8: Longitude
			"""
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
			norm_temp = normalize(data_temp[:,1:9], axis=0, norm='max')
			data_temp[:,1:9] = norm_temp #normalize all but the storm id
			unique_storms = np.unique(data_temp[:,0])
			data = np.empty((len(unique_storms), 337, 9))
			data[:] = self.pad_type
			for i, storm_id in enumerate(unique_storms):
				storm_data = data_temp[data_temp[:,0]==storm_id, 1:]
				pad_width = 337 - len(storm_data) 
				storm_data = np.pad(storm_data, ((pad_width,0), (0,0)), 
							'constant', constant_values=self.pad_type)
				data[i] = storm_data
			if ts_delete_step_size != 0:
				data = data[:, 0::ts_delete_step_size+1, :]

			self.storms = unique_storms.astype(int)

		elif ts_input == False:
			""" - if not reading time series as input, each storm has an input
					with dimentions of 1x4. 
				- input values are as follows:
					0: TrackID
					1: Central Pressure Deficit
					2: Radius to Max Winds
					3: Translation Speed
			"""
			path_to_data = os.path.join(path_to_data, 'StormConditions.csv')
			df = pd.read_csv(path_to_data)
			df = df[~df['StormID'].isin(missing_storms[0])]
			self.storms = df['StormID'].astype(int).values
			data = df[['TrackID', 'CentralPressureDeficit', 
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
		df.fillna(0, inplace=True)		# note: want to confirm this with team
		data = df.values
		self.sp_list = sp_list
		return data

	def read_data_ts(self, path_to_data, savepoints, storm):
		path_to_data = os.path.join(path_to_data, 'CHS_Storms_raw')
		file_name = ('NACCS_TP_{0:04d}_SYN_Tides_0_SLC_0_RFC_0_surge_all.mat' 
							.format(storm))
		mat_file = os.path.join(path_to_data, file_name)
		data = io.loadmat(mat_file)['surge']	# reading in data
		data = data[:,savepoints-1]			# isolating savepoints in dataset

		""" padding data with nan values so that all time series are the same 
			size. the value of 1980 is used b/c it's the largest of all time
			series """
		pad_width = 1980 - np.shape(data)[0] 
		data = np.pad(data, ((0,pad_width), (0,0)), 
					'constant', constant_values=self.pad_type)
		return data

	def __len__(self):
		return len(self.storm_conds)

	def __getitem__(self, idx):
		data_val = self.storm_conds[idx]
		if self.ts_output == False:
			target = self.target[idx]
		else:
			target = self.read_data_ts(path_to_data, self.savepoints, self.storms[idx])

		return data_val, target

if __name__ == "__main__":
	# path to data
	path_to_data = os.path.join(os.getcwd(), '..', 'data')
	
	""" defining bounding box """
	# small bounding box
	xmin, xmax = -74.2754, -73.9374
	ymin, ymax = 40.4041, 40.6097
	
	train_test_split = 0.8		# ratio to split test and train data

	# dataset class
	dataset = CHS_DataSet(path_to_data, xmin, xmax, ymin, ymax, ts_input=True,
		ts_output=False, ts_delete_step_size=0)
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



