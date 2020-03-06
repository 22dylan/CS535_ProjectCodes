"""
this code takes DOGAMI tsunami output files as .DAT and converts them to raster files.
there are multiple output files that are gridded across the entire oregon coast,
	however (i presume) that there are finer grids at each respective location (e.g. "South Coast")

Note that the tillamook grid came with two sets of output files: (1) tillamook and (2) NN
	-the maps of the tsunami project areas have three grids in Tillamook county: (1) NN_South, 
		(2) Tillamook, and (3) NN_North
	-DRS made a copy of NN, renamed one NN_North and the other NN_South. 
"""

import os
import numpy as np
import pandas as pd
import gdal

storm_StartEnd = [1, 1030]

path_to_data = os.path.join(os.getcwd(), '..', 'data', 'Max_Surge_nyc', 'CSVs')
max_storm_files = os.listdir(path_to_data)

file_consider = ['TS_{}_MaxSurge.csv' .format(i) for i in range(storm_StartEnd[0], 
								storm_StartEnd[1]+1)]

pathout = os.path.join(os.getcwd(), '..', 'data', 'Max_Surge_nyc', 'Rasters')
xmin = -74.619
xmax = -73.397
ymin = 40.080
ymax = 40.892


for file in file_consider:

	if file not in max_storm_files:
		print('File {} not available.' .format(file))
		continue

	print('Processing: {}' .format(file))
	vrt_fn = file.replace('.csv', '.vrt')	
	vrt_fn = os.path.join(pathout, 'VRTs', vrt_fn)
	lyr_name = file.replace('.csv', '')
	out_tif = file.replace('.csv', '.tiff')
	outfilename = os.path.join(pathout, out_tif)


	file = os.path.join(path_to_data, file)

	with open(vrt_fn, 'w') as fn_vrt:
		fn_vrt.write('<OGRVRTDataSource>\n')
		fn_vrt.write('\t<OGRVRTLayer name="%s">\n' % lyr_name)
		fn_vrt.write('\t\t<SrcDataSource>%s</SrcDataSource>\n' % file)
		fn_vrt.write('\t\t<GeometryType>wkbPoint</GeometryType>\n')
		fn_vrt.write('\t\t<LayerSRS>WGS84</LayerSRS>\n')
		fn_vrt.write('\t\t<GeometryField encoding="PointFromColumns" x="SP_Longitude" y="SP_Latitude" z="Z_max"/>\n')

		fn_vrt.write('\t</OGRVRTLayer>\n')
		fn_vrt.write('</OGRVRTDataSource>\n')

	width = 2000
	height = 2000
	rad_x = (xmax - xmin)/50
	rad_y = (ymax - ymin)/50

	# algorithm_str = 'invdist:power=3:radius1={}:radius2={}:max_points=20:min_points=5' .format(rad_x, rad_y)
	# algorithm_str = 'invdist:power=3:max_points=20:min_points=15'
	algorithm_str = 'invdistnn'

	options = gdal.GridOptions(width=width, height=height, noData=-9999, algorithm=algorithm_str)
	output = gdal.Grid(outfilename,vrt_fn,options=options)
		
	fds
# ---------------

