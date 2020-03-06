import os
import pandas as pd
import numpy as np
import scipy.io as io


class switch_sp_and_storm():
	def __init__(self):
		self.path_to_storms = os.path.join(os.getcwd(), '..', '..', 'data', 
					'CHS_Storms_raw')
		self.path_out = os.path.join(os.getcwd(), '..', '..', 'data', 
					'CHS_Storms_reorganized')
		self.all_storm_files = os.listdir(self.path_to_storms)
		self.savepoints = list(range(1, 18978))

	def create_empty_savepoint_files(self):
		""" OVERWRITES WHAT'S IN FILES
			creating empty savepoint files
		"""
		print('positive you want to continue??')
		FDSFADSFDSF	# TO BREAK THE CODE IF TRYING TO RUN

		# for sp in self.savepoints:
		# 	if (sp%100) == 0:
		# 		print(sp)
		# 	df = pd.DataFrame()
		# 	df['time'] = np.arange(0, (500+1)*10, 10)
		# 	filename = os.path.join(self.path_out, 'SP_{}.csv' .format(sp))
		# 	df.to_csv(filename, index=False)

	def write_storm_data(self):
		# writing storm data to savepoint files
		for storm in self.all_storm_files:
			stormfile = os.path.join(self.path_to_storms, storm)
			data = io.loadmat(stormfile)['surge']	# reading in data
			storm_num = int(storm.split('_')[2])
			print('Processing: {}' .format(storm))

			for sp in self.savepoints:
				if (sp%100) == 0:
					print('\tsp: {}' .format(sp))


				ts = data[:,sp-1]
				sp_file = os.path.join(self.path_out, 'SP_{}.csv' .format(sp))
				df = pd.read_csv(sp_file)
				df1 = pd.DataFrame()
				df1['storm_{}' .format(storm_num)] = ts
				if len(df1) > len(df):
					df.drop('time', axis=1, inplace=True)
					time = pd.DataFrame()
					time['time'] = np.arange(0, (len(df1+1))*10, 10)
					df = pd.concat([df, time], ignore_index=False, axis=1)

				df = pd.concat([df,df1], ignore_index=False, axis=1)
				df.to_csv(sp_file, index=False)


if __name__ == "__main__":
	sp_st = switch_sp_and_storm()
	sp_st.write_storm_data()