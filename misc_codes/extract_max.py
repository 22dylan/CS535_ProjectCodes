import os
import numpy as np
import pandas as pd
import scipy.io as io


path_to_data = os.path.join(os.getcwd(), '..', '..','data', 'CHS_Storms_raw')
path_out = os.path.join(os.getcwd(), '..', '..', 'data', 'Max_Surge2', 'CSVs')
path_to_conversion = os.path.join(os.getcwd(), '..', '..', 'data', 
		'NACCS_SavePts_18977_ConversionKey_MSL_to_NAVD_meters.csv')
LatLon_key = pd.read_csv(path_to_conversion, skiprows=[1])


all_files = os.listdir(path_to_data)
storms = [int(i.split('_')[2]) for i in all_files]

completed_runs = os.listdir(path_out)
processed_storms = [int(i.split('_')[1]) for i in completed_runs]

missing_storms = np.setdiff1d(storms, processed_storms)

for i, storm in enumerate(missing_storms):

	file_name = ('NACCS_TP_{0:04d}_SYN_Tides_0_SLC_0_RFC_0_surge_all.mat' 
						.format(storm))

	file_path = os.path.join(path_to_data, file_name)
	print('Storm: {}' .format(storm))

	if storm == 299:
		print('\tpassed')
		continue

	data = io.loadmat(file_path)['surge']	# reading in data

	# max_surge = np.amax(data, axis=0)		# taking max
	max_surge = np.nanmax(data, axis=0)
	max_surge[max_surge<0] = np.nan			# replacing NaNs
	print(max_surge[0:10])
	

	# writing out results
	df_out = pd.DataFrame()
	df_out['SavePointID'] = LatLon_key['SavePointID']
	df_out['SP_Latitude'] = LatLon_key['SP_Latitude']
	df_out['SP_Longitude'] = LatLon_key['SP_Longitude']
	df_out['MSL_to_NAVD88'] = LatLon_key['MSL_to_NAVD88']
	df_out['Z_max'] = max_surge

	file_out = 'TS_{}_MaxSurge.csv' .format(storm)
	file_out = os.path.join(path_out, file_out)
	df_out.to_csv(file_out, index=False)
