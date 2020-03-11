import os 
import pandas as pd
import numpy as np
import re

def natural_sort(l): 
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
	return sorted(l, key = alphanum_key)


path_to_data = os.path.join(os.getcwd(), '..', '..', 'data', 'Max_Surge2', 
							'CSVs')
files = natural_sort(os.listdir(path_to_data))

d_out = np.zeros((len(files), 18978))
storm_id_list = []
i = 0
for file in files:
	print(file)
	full_path = os.path.join(path_to_data, file)
	df = pd.read_csv(full_path)
	storm_id = file.split('_')[1]
	d_out[i, 1:] = df['Z_max']
	storm_id_list.append(storm_id)
	i+=1
	
d_out[:,0] = storm_id_list

header = ['Storm_ID']
for i in range(18977):
	header.append('sp_{}' .format(i+1))

df = pd.DataFrame(d_out, columns=header)
path_out = os.path.join(os.getcwd(), '..', '..', 'data', 'Max_Surge2', 'max_surge2.csv')
df.to_csv(path_out, index=False)
