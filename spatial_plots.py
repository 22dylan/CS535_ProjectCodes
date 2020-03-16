import os
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import contextily as ctx

mpl.use('TkAgg')

"""
script to plot average error at all save points
"""

sp2coord = os.path.join(os.getcwd(), '..', 'data', 
				'NACCS_SavePts_18977_ConversionKey_MSL_to_NAVD_meters.csv')
sp2coord = pd.read_csv(sp2coord, skiprows=[1])
sp2coord['SP_Longitude'] = pd.to_numeric(sp2coord['SP_Longitude'])
sp2coord['SP_Latitude'] = pd.to_numeric(sp2coord['SP_Latitude'])

path_to_models = os.path.join(os.getcwd(), 'LSTM_training_results', 
						'Model_results')
models = [
			"LSTM_B15_h100_l1_bbS",
			"LSTM_LL1_B15_h100_l1_bbS",
			"LSTM_LL2_B15_h100_l1_bbS",
			
			"LSTM_LL1_B15_h100_l1_bbM",
			"LSTM_LL2_B15_h100_l1_bbM",
			
			"LSTM_LL1_B15_h100_l1_bbL",
			"LSTM_LL2_B15_h100_l1_bbL",
			  ]

model_names = [
			'LSTM',
			'LSTM + Linear Layer',
			'LSTM + Norm Layer + Linear Layer',

			'LSTM + Linear Layer',
			'LSTM + Norm Layer + Linear Layer',

			'LSTM + Linear Layer',
			'LSTM + Norm Layer + Linear Layer'
   			]

bb = ['Small', 'Small', 'Small', 
	  'Medium', 'Medium', 
	  'Large', 'Large' ]

cbar_axes = [
			[0.92, 0.105, 0.02, 0.78],		# small bounding box
			[0.92, 0.105, 0.02, 0.78],
			[0.92, 0.105, 0.02, 0.78],

			[0.92, 0.25, 0.02, 0.5],		# med bounding box
			[0.92, 0.25, 0.02, 0.5],
			
			[0.83, 0.105, 0.02, 0.78],		# lg. bounding box
			[0.83, 0.105, 0.02, 0.78]
			]
cbar_max = [0.35, 0.35, 0.35, 
			0.7, 0.7,
			0.4, 0.4
			]

for i, model in enumerate(models):
	predict = model + '_predict.csv'
	predict = os.path.join(path_to_models, predict)
	predict = pd.read_csv(predict).mean(axis=0)

	target = model + '_target.csv'
	target = os.path.join(path_to_models, target)
	target = pd.read_csv(target).mean(axis=0)

	error = predict-target
	
	sp = error.index.to_list()	
	sp = [(i.split('_')[1]) for i in sp]
	sp2coord_temp = sp2coord[sp2coord['SavePointID'].isin(sp)]
	sp2coord_temp['error'] = error.values

	gdf = gpd.GeoDataFrame(sp2coord_temp, 
    					   geometry=gpd.points_from_xy(
    					   			sp2coord_temp.SP_Longitude, 
    					   			sp2coord_temp.SP_Latitude
    					   			)
    					   )

	gdf.crs = {'init': 'epsg:4326'}
	gdf = gdf.to_crs({'init': 'epsg:3857'})

	fig, axes = plt.subplots(1,1, figsize = (12, 10))
	gdf.plot(column=error, cmap = 'bwr', vmin=-cbar_max[i], vmax=cbar_max[i], 
			ax=axes, legend = False, edgecolor = 'k', linewidth = 0.3)
	ctx.add_basemap(axes, url=ctx.providers.Stamen.TonerLite)
	axes.set_xticks([])
	axes.set_yticks([])
	axes.set_title('Model: {}\nBounding Box: {}' .format(model_names[i], bb[i]))
	
	cbar_ax = fig.add_axes(cbar_axes[i])
	norm = mpl.colors.Normalize(vmin=-cbar_max[i],vmax=cbar_max[i])
	
	sm = plt.cm.ScalarMappable(cmap='bwr', norm=norm)
	sm.set_array([])
	plt.colorbar(sm, cax=cbar_ax)

	path_out = os.path.join(os.getcwd(), 'spatial_plots', model)
	plt.savefig(path_out)


























