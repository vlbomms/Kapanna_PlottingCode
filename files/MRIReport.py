import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from numpy.ma import masked_array
import csv as _csv
import os as _os
import sys as _sys
import glob
from datetime import datetime

import json
import nrrd

import plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import scipy
from scipy import stats

from pygam import ExpectileGAM
from pygam.datasets import mcycle
from pygam import LinearGAM

from IPython.display import HTML as idhtml
from weasyprint import HTML as wp

# Script to calculate spare AD score for our test subject
from spareAD import calculateSpareAD, createSpareADplot
from spareBA import calculateSpareBA, createSpareBAplot

# Script to create cmap
from createcmap import get_continuous_cmap

### Stored file constants
MUSE_ROI_Mapping = '/refs/MUSE_DerivedROIs_Mappings.csv'
maphemi = pd.read_csv('/refs/MUSE_ROI_Dictionary.csv')
MUSE_Ref_Values = '/refs/PENNcombinedharmonized_out.csv'

################################################ FUNCTIONS ################################################

##########################################################################
##### Function to read demographic info from json
def readSubjInfo(subjfile):
	
	if subjfile.endswith('.csv'):
		subjAge = None
		subjSex = None
		if subjfile is not None:
			d={}
			with open(subjfile) as f:
				reader = _csv.reader( f )
				subDict = {rows[0]:rows[1] for rows in reader}
	
	if subjfile.endswith('.json'):
		with open(subjfile, 'r') as read_file:
			data = json.load(read_file)
			
			## WARNING: Trying to parse fields in a robust way
			## This part is based on assumptions on dicom field names
			## it should be tested carefully and updated for different patterns
			subID = [value for key, value in data.items() if 'PatientID' in key][0]
			
			for key, value in data.items():
				if ('PatientAge' in key) and ('Y' in str(value)):
					print("Successfully identified patient age in JSON...")
					subAge = float(str(value).replace('Y',''))
					break

			subSex = [value for key, value in data.items() if 'PatientSex' in key][0]
			subExamDate = [value for key, value in data.items() if 'StudyDate_date' in key][0]
			subExamDate = datetime.strptime(subExamDate, "%Y-%m-%d").strftime("%m/%d/%Y")
			
			subDict = {'MRID':subID, 'Age':subAge, 'Sex':subSex, "ExamDate":subExamDate}

	return subDict

##########################################################################
##### Function to calculate brain mask volume
def calcMaskVolume(maskfile):
	
	### Check input mask
	if not maskfile:
		print("ERROR: Input file not provided!!!")
		sys.exit(0) 

	### Read the input image
	roinii = nrrd.read_header(maskfile)
	roiimg = nrrd.read(maskfile)[0]

	### Get voxel dimensions
	voxdims1, voxdims2, voxdims3 = roinii['pixdim[1]'], roinii['pixdim[2]'], roinii['pixdim[3]']
	voxvol = float(voxdims1)*float(voxdims2)*float(voxdims3)

	### Calculate mask volume
	maskVol = voxvol * np.sum(roiimg.flatten()>0)
	
	return maskVol

### Function to calculate derived ROI volumes for MUSE
def calcRoiVolumes(maskfile, mapcsv, dfRef, subDict):
	
	### Check input mask
	if not maskfile:
		print("ERROR: Input file not provided!!!")
		sys.exit(0) 

	### Read the input image
	roinii = nrrd.read_header(maskfile)
	roiimg = nrrd.read(maskfile)[0]

	### Get voxel dimensions
	voxdims1, voxdims2, voxdims3 = roinii['pixdim[1]'], roinii['pixdim[2]'], roinii['pixdim[3]']
	voxvol = float(voxdims1)*float(voxdims2)*float(voxdims3)

	### Calculate ROI count and volume
	ROIs, Counts = np.unique(roiimg, return_counts=True)
	ROIs = ROIs.astype(int)
	Volumes = voxvol * Counts

	### Create an array indexed from 0 to max ROI index
	###   This array will speed up calculations for calculating derived ROIs
	###   Instead of adding ROIs in a loop, they are added at once using: 
	###          np.sum(all indexes for a derived ROI)  (see below)
	VolumesInd = np.zeros(ROIs.max()+1)
	VolumesInd[ROIs] = Volumes

	### Calculate derived volumes
	DerivedROIs = []
	DerivedVols = []
	ROIlist = []
	Vollist = []
	with open(mapcsv) as mapcsvfile:
		reader = _csv.reader(mapcsvfile, delimiter=',')

		# Read each line in the csv map files
		for row in reader:
			row = list(filter(lambda a: a != '', row))
			# Append the ROI number to the list
			DerivedROIs.append(row[0])
			roiInds = [int(x) for x in row[2:]]
			DerivedVols.append(np.sum(VolumesInd[roiInds]))

	### Decide what to output, should fix post-presentation
	all_MuseROIs =  dict(zip(DerivedROIs, DerivedVols))
	dictr = {}
	allz = {}
	allz_num = {}
	all_MuseROIs_name = {}

	# Get all left and right ROIs
	Rinds = list(maphemi.loc[(maphemi['HEMISPHERE'] == 'R'), 'ROI_INDEX'].values)
	Linds = list(maphemi.loc[(maphemi['HEMISPHERE'] == 'L'), 'ROI_INDEX'].values)

	#### Correct ref values temporarily for age + gender controlled z-score calculation ####
	nonROI = list(dfRef.columns[dfRef.columns.str.contains('AI')].values)
	nonROI.extend(['MRID','Study','PTID','Age','Sex','Diagnosis_nearest_2.0','SITE','Date','ICV','SPARE_AD','SPARE_BA'])
	dfRefTmp = dfRef.copy(deep=True)
	# Rename 702 to ICV
	dfRefTmp.rename(columns={'702':'ICV'}, inplace=True)

	# Get precise age and sex references only
	dfRefTmp = dfRefTmp[dfRefTmp.Sex == subDict['Sex']]
	dfRefTmp['Age'] = dfRefTmp['Age'].round(0)
	dfRefTmp = dfRefTmp[dfRefTmp.Age == round(subDict['Age'],0)]

	# Actually do the ICV-adjust
	dfRefTmp[dfRefTmp.columns.difference(nonROI)] = dfRefTmp[dfRefTmp.columns.difference(nonROI)].div(dfRefTmp['ICV'], axis=0)*dfRefTmp['ICV'].mean()

	# Convert from mm^3 to cm^3
	dfRefTmp[dfRefTmp.columns.difference(nonROI)] = dfRefTmp[dfRefTmp.columns.difference(nonROI)]/1000

	## Create anatomical structure to z-score equivalency for the subject; only considering ROIs Ilya said to use!
	for i in list(all_MuseROIs.keys()):
		if int(i) <= 207 and int(i) not in [j for j in range(81,89)] and int(i) not in [4,11,51,52]:
			allz[i] = ((all_MuseROIs[i]/1000) - dfRefTmp[i].mean())/(dfRefTmp[i].std())
			allz_num[i] = allz[i]
			name = list(maphemi.loc[(maphemi['ROI_INDEX'] == int(i)), 'ROI_NAME'].values)[0]
			allz[name] = allz.pop(i)
			all_MuseROIs_name[name] = all_MuseROIs[i]
		else:
			continue
	
	### Select patient ROIs to report and update both patient and reference ROIs to colloquial terminology ###

	### Brain volumes ###
	# Total brain volume
	dictr["Total Brain Volume"] = all_MuseROIs["701"]
	dfRef.rename(columns={"701":"Total Brain Volume"}, inplace = True)
	# Right brain volume (1/2 of all indivisible ROIs: 4,11,35,46,71,72,73,95)
	dictr["Right Brain Volume"] = np.sum(VolumesInd[Rinds]) + np.sum(VolumesInd[[4,11,35,71,72,73,95]])/2
	dfRef["Right Brain Volume"] = dfRef[[str(i) for i in Rinds]].sum(axis=1) + (dfRef[['4','11','35','71','72','73','95']].sum(axis=1))/2
	# Left brain volume (1/2 of all indivisible ROIs: 4,11,35,46,71,72,73,95)
	dictr["Left Brain Volume"] = np.sum(VolumesInd[Linds]) + np.sum(VolumesInd[[4,11,35,71,72,73,95]])/2
	dfRef["Left Brain Volume"] = dfRef[[str(i) for i in Linds]].sum(axis=1) + (dfRef[['4','11','35','71','72','73','95']].sum(axis=1))/2
	# Asymmetry index for brain
	dictr["Brain Volume AI"] = (dictr["Left Brain Volume"] - dictr["Right Brain Volume"])/((dictr["Left Brain Volume"] + dictr["Right Brain Volume"])/2)
	dfRef["Brain Volume AI"] = (dfRef["Left Brain Volume"] - dfRef["Right Brain Volume"]).div((dfRef["Left Brain Volume"] + dfRef["Right Brain Volume"])/2,axis=0)

	### Brainstem ### - had to adjust this ROI
	dictr["Total Brainstem"] = all_MuseROIs["35"] + all_MuseROIs["61"] + all_MuseROIs["62"]
	dfRef["35"] = dfRef["35"] + dfRef["61"] + dfRef["62"]
	dfRef.rename(columns={"35":"Total Brainstem"}, inplace = True)

	### Ventricles###
	# Total ventricle
	dictr["Total Ventricle"] = all_MuseROIs["509"]
	dfRef.rename(columns={"509":"Total Ventricle"}, inplace = True)
	# Right ventricle (1/2 of 3rd + 4th)
	dictr["Right Ventricle"] = all_MuseROIs["49"] + all_MuseROIs["51"] + all_MuseROIs["4"]/2 + all_MuseROIs["11"]/2
	dfRef["Right Ventricle"] = dfRef["49"] + dfRef["51"] + dfRef["4"]/2 + dfRef["11"]/2
	# Left ventricle (1/2 of 3rd + 4th)
	dictr["Left Ventricle"] = all_MuseROIs["50"] + all_MuseROIs["52"] + all_MuseROIs["4"]/2 + all_MuseROIs["11"]/2
	dfRef["Left Ventricle"] = dfRef["50"] + dfRef["52"] + dfRef["4"]/2 + dfRef["11"]/2
	# Asymmetry index for ventricles
	dictr["Ventricle AI"] = (dictr["Left Ventricle"] - dictr["Right Ventricle"])/((dictr["Left Ventricle"] + dictr["Right Ventricle"])/2)
	dfRef["Ventricle AI"] = (dfRef["Left Ventricle"] - dfRef["Right Ventricle"]).div((dfRef["Left Ventricle"] + dfRef["Right Ventricle"])/2,axis=0)

	### Gray Matter ###
	# Total gray matter
	dictr["Total Gray Matter"] = all_MuseROIs["601"]
	dfRef["601"] = dfRef["601"]
	dfRef.rename(columns={"601":"Total Gray Matter"}, inplace = True)
	# Right gray matter - had to adjust this ROI with 1/2 of certain ROIs
	dictr["Right Gray Matter"] = all_MuseROIs["613"] + all_MuseROIs["71"]/2 + all_MuseROIs["72"]/2 + all_MuseROIs["73"]/2
	dfRef["613"] = dfRef["613"] + dfRef["71"]/2 + dfRef["72"]/2 + dfRef["73"]/2
	dfRef.rename(columns={"613":"Right Gray Matter"}, inplace = True)
	# Left gray matter - had to adjust this ROI with 1/2 of certain ROIs
	dictr["Left Gray Matter"] = all_MuseROIs["606"] + all_MuseROIs["71"]/2 + all_MuseROIs["72"]/2 + all_MuseROIs["73"]/2
	dfRef["606"] = dfRef["606"] + dfRef["71"]/2 + dfRef["72"]/2 + dfRef["73"]/2
	dfRef.rename(columns={"606":"Left Gray Matter"}, inplace = True)

	# Asymmetry index for gray matter
	dictr["Gray Matter AI"] = (dictr["Left Gray Matter"] - dictr["Right Gray Matter"])/((dictr["Left Gray Matter"] + dictr["Right Gray Matter"])/2)
	dfRef["Gray Matter AI"] = (dfRef["Left Gray Matter"] - dfRef["Right Gray Matter"]).div((dfRef["Left Gray Matter"] + dfRef["Right Gray Matter"])/2,axis=0)

	### White Matter ###
	# Total white matter
	dictr["Total White Matter"] = all_MuseROIs["604"]
	dfRef.rename(columns={"604":"Total White Matter"}, inplace = True)
	# Right white matter - had to adjust this ROI with 1/2 of certain ROIs
	dictr["Right White Matter"] = all_MuseROIs["614"] + all_MuseROIs["95"]/2
	dfRef.rename(columns={"614":"Right White Matter"}, inplace = True)
	# Left white matter - had to adjust this ROI with 1/2 of certain ROIs
	dictr["Left White Matter"] = all_MuseROIs["607"] + all_MuseROIs["95"]/2
	dfRef.rename(columns={"607":"Left White Matter"}, inplace = True)
	# Asymmetry index for white matter
	dictr["White Matter AI"] = (dictr["Left White Matter"] - dictr["Right White Matter"])/((dictr["Left White Matter"] + dictr["Right White Matter"])/2)
	dfRef["White Matter AI"] = (dfRef["Left White Matter"] - dfRef["Right White Matter"]).div((dfRef["Left White Matter"] + dfRef["Right White Matter"])/2, axis = 0)

	### Hippocampus ###
	# Total Hippocampus
	dictr["Total Hippocampus"] = all_MuseROIs["47"] + all_MuseROIs["48"]
	dfRef["Total Hippocampus"] = dfRef["47"] + dfRef["48"]
	# Right Hippocampus
	dictr["Right Hippocampus"] = all_MuseROIs["47"]
	dfRef["Right Hippocampus"] = dfRef["47"]
	#Left Hippocampus
	dictr["Left Hippocampus"] = all_MuseROIs["48"]
	dfRef["Left Hippocampus"] = dfRef["48"]
	# Asymmetry index for hippocampus
	dictr["Hippocampus AI"] = (dictr["Left Hippocampus"] - dictr["Right Hippocampus"])/((dictr["Left Hippocampus"] + dictr["Right Hippocampus"])/2)
	dfRef["Hippocampus AI"] = (dfRef["Left Hippocampus"] - dfRef["Right Hippocampus"]).div((dfRef["Left Hippocampus"] + dfRef["Right Hippocampus"])/2, axis = 0)

	return dictr, dfRef, allz, allz_num, all_MuseROIs, all_MuseROIs_name

def plotWithRef(dfRef, dfSub, selVarlst, fname, spareAD, spareBA):
	selVarlst.append("SPARE_AD")
	selVarlst.append("SPARE_BA")
	## TODO: make specs arg. automatically adjusted to number of plots
	fig = py.subplots.make_subplots(rows=3,cols=3,subplot_titles=selVarlst,specs=[[{}, {}, {}], [{}, {}, {}], [{}, {}, None]],vertical_spacing = 0.05,horizontal_spacing = 0.03)
	row = 1
	column = 1
	sl = False
	mark = True

	# Get only those reference subjects ±5 years from subject age
	lowlim = int(dfSub.Age.values[0]) - 5
	uplim = int(dfSub.Age.values[0]) + 5

	ADRef = dfRef.loc[dfRef['Diagnosis_nearest_2.0'] == 'AD']
	CNRef = dfRef.loc[dfRef['Diagnosis_nearest_2.0'] == 'CN']

	for selVar in selVarlst[:-2]:
		## Only allow legend to show up for one of the plots (for display purposes)
		if mark:
			sl = True
			mark = False
		else:
			sl = False

		X_CN = CNRef.Age.values.reshape([-1,1])
		y_CN = CNRef[selVar].values.reshape([-1,1])

		X_AD = ADRef.Age.values.reshape([-1,1])
		y_AD = ADRef[selVar].values.reshape([-1,1])

		##############################################################
		########### CN values ###########
		# fit the mean model first by CV
		gam50 = ExpectileGAM(expectile=0.5).gridsearch(X_CN, y_CN)

		# and copy the smoothing to the other models
		lam = gam50.lam

		# fit a few more models
		gam90 = ExpectileGAM(expectile=0.90, lam=lam).fit(X_CN, y_CN)
		gam10 = ExpectileGAM(expectile=0.10, lam=lam).fit(X_CN, y_CN)
	
		XX_CN = gam50.generate_X_grid(term=0, n=100)
		XX90_CN = list(gam90.predict(XX_CN).flatten())
		XX50_CN = list(gam50.predict(XX_CN).flatten())
		XX10_CN = list(gam10.predict(XX_CN).flatten())
		XX_CN = list(XX_CN.flatten())

		########### AD values ###########
		# fit the mean model first by CV
		gam50 = ExpectileGAM(expectile=0.5).gridsearch(X_AD, y_AD)

		# and copy the smoothing to the other models
		lam = gam50.lam

		# fit a few more models
		gam90 = ExpectileGAM(expectile=0.90, lam=lam).fit(X_AD, y_AD)
		gam10 = ExpectileGAM(expectile=0.10, lam=lam).fit(X_AD, y_AD)
	
		XX_AD = gam50.generate_X_grid(term=0, n=100)
		XX90_AD = list(gam90.predict(XX_AD).flatten())
		XX50_AD = list(gam50.predict(XX_AD).flatten())
		XX10_AD = list(gam10.predict(XX_AD).flatten())
		XX_AD = list(XX_AD.flatten())

		fig.append_trace( go.Scatter(mode='markers', x=ADRef.Age.tolist(), y=ADRef[selVar].tolist(), legendgroup='AD', marker=dict(color='MediumVioletRed', size=5), opacity=0.2, name='AD Reference', showlegend=sl),row,column)
		fig.append_trace( go.Scatter(mode='markers', x=CNRef.Age.tolist(), y=CNRef[selVar].tolist(), legendgroup='CN', marker=dict(color='MediumBlue', size=5), opacity=0.2, name='CN Reference', showlegend=sl),row,column)
		fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX90_CN, legendgroup='CN', marker=dict(color='MediumPurple', size=10), name='90th percentile for CN', showlegend=sl),row,column)
		fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX50_CN, legendgroup='CN', marker=dict(color='MediumVioletRed', size=10), name='50th percentile for CN', showlegend=sl),row,column)
		fig.append_trace( go.Scatter(mode='lines', x=XX_CN, y=XX10_CN, legendgroup='CN', marker=dict(color='MediumBlue', size=10), name='10th percentile for CN', showlegend=sl),row,column)
		fig.append_trace( go.Scatter(mode='lines', x=XX_AD, y=XX90_AD, legendgroup='AD', marker=dict(color='MediumPurple', size=10), name='90th percentile for AD', line=dict(dash = 'dash'), showlegend=sl),row,column)
		fig.append_trace( go.Scatter(mode='lines', x=XX_AD, y=XX50_AD, legendgroup='AD', marker=dict(color='MediumVioletRed', size=10), name='50th percentile for AD', line=dict(dash = 'dash'), showlegend=sl),row,column)
		fig.append_trace( go.Scatter(mode='lines', x=XX_AD, y=XX10_AD, legendgroup='AD', marker=dict(color='MediumBlue', size=10), name='10th percentile for AD', line=dict(dash = 'dash'), showlegend=sl),row,column)
		fig.append_trace( go.Scatter(mode='markers', x=dfSub.Age.tolist(), y=dfSub[selVar].tolist(), legendgroup='Patient', marker=dict(color='Black', symbol = 'circle-cross-open', size=16,line=dict(color='MediumPurple', width=3)), name='Patient', showlegend=sl),row,column)

		fig.update_xaxes(range=[lowlim, uplim])	

		## Allow iteration through nx3 structure of plots
		if column == 3:
			row += 1
			column -= 2
		else:
			column += 1

	## Get the word 'Age' underneath each plot
	#for i in range(1,len(selVarlst)+1):
	#	fig['layout']['xaxis{}'.format(i)]['title']='Age'

	fig = createSpareADplot(spareAD, dfSub, fig, 3, fname)
	createSpareBAplot(spareBA, dfSub, fig, 3, fname)
	
## Write the HTML code for display --- TODO: Move to seperate python file
def writeHtml(plots, tables, flagtable, dftable, outName, brainplot):
	all_plot = ""
	all_brainplot = ""
	all_table = ""
	all_flag = ""

	#Generating HTML for plots and tables
	for plot in plots:
		ind = '''<img src=''' + plot + '''></img>'''
		all_plot = all_plot + ind
	for table in tables:
		ind = '' + table + ''
		all_table = all_table + ind
	for plot in brainplot:
		ind = '''<img src=''' + plot + ''' class="center"></img>'''
		all_brainplot = all_brainplot + ind
	for table in flagtable:
		ind = '' + table + ''
		all_flag = all_flag + ind

	#HTML code for report structure
	html_string_test = '''
	<html>
	<head>
		<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
		<style>
			body { margin:0 0; #background:whitesmoke; }
			.toohigh { background: #ADD8E6 !important; color: white; }
			.toolow { background: #FFB6C1 !important; color: white; }
			.norm { background: white !important; color: white; }
			h6 { margin: 1em 0 0.5em 0 ! important; 
				font-weight: normal ! important;
				position: relative ! important;
				text-shadow: 0 -1px rgba(0,0,0,0.6) ! important;
				background: #355681 ! important;
				background: rgba(53,86,129, 0.8) ! important;
				border: 1px solid #fff ! important;
				padding: 5px 15px ! important;
				color: white ! important;
				border-radius: 0 10px 0 10px ! important;
				box-shadow: inset 0 0 5px rgba(53,86,129, 0.5) ! important;
				font-family: 'Muli', sans-serif ! important; 
				}
		</style>
		<style type="text/css">
			@page { margin: 45px 0px; }
  			td {
    			padding: 0 5px;
    			text-align: center;
    			max-width: 100%;
  				}
  			th {
  				text-align: center;
  				max-width: 100%;
  				}
  			.header{
  			position: fixed;
    		left: 0px;
    		right: 0px;
    		height: 42px;
  			top: -45px;
  			bottom: 0px;
  			background-color: #D3D3D3 ! important;
  			}
  			.footer{
  			position: fixed;
  			left: 5px;
  			right: 5px;
  			bottom: -32px;
  			height: 30px;
  			}
  			.p{
  			text-align: center ! important;
			vertical-align: middle ! important;
			margin-top:10px ! important;
			margin-left:30px ! important
			font-size: 10px ! important;
  			}
  			.center {
  			display: block;
  			margin-left: auto;
  			margin-right: auto;
  			}
		</style>
		<style>
			table {
				border-collapse: collapse;
  				table-layout: auto;
    			width: 100%;
				}
			tr {
  				border: solid;
  				border-width: 1px 0;
  				max-width: 100%;
				}
		</style>
	</head>
	<body>
		<div class="header";>
			<div class="col-xs-4">
				<p>''' + dftable.loc[0].values[0] + ' | ' + str(int(float(dftable.loc[1].values[0]))) + 'y' + ' ' + dftable.loc[2].values[0] + '''<br>Scan Date: ''' + dftable.loc[3].values[0] + '''</p>
			</div>
			<div class="col-xs-4">
				<img src="/logos/cbica.png">
			</div>
			<div class="col-xs-4">
				<img src="/logos/PennMedicineLogo.png">
			</div>
		</div>
		<div class="container">
			<h6>Age plots: </h6>
			''' + all_plot + '''
		</div>
		<div class="container">
			<h6>Brain structures: </h6>
			''' + all_brainplot + '''
		</div>
		<div class="container">
			<h6>Brain structure volumes: </h6>
			''' + all_table + '''
		</div>
		<div class="container">
			<h6>Flagged volumes: </h6>
			''' + all_flag + '''
		</div>
		<div class="footer">
			<p>Note: These measurements may be useful as an adjunct to other diagnostic evaluations and the broader clinical context. Measurements do not diagnose a specific underlying disease in isolation.</p>
		</div>
	</body>
	</html>'''

	f = open(outName,'w')
	f.write(html_string_test)
	f.close()

# Create color coding system based on standard deviations (-1.28 and 1.28 stds are 10th and 90th percentile, respectively)
def tagID(num):
	if type(num) != str:
		if num >= 1.28:
			tag = "toohigh"
		elif num <= -1.28:
			tag = "toolow"
		else:
			tag = "norm"
	else:
		return "norm"

	return tag

## Create table with volume values + percentiles
def makeTables(dftable):
	all_entries = ""
	
	for index, rows in dftable.iterrows():
		tbody = ""
		tbody = '''<tr style="text-align: center;"><td><b>''' + rows["Brain Region"] + '''</b></td><td>''' + rows["Volume (cubic mL)"] + '''</td>''' + '''<b><td>''' + rows["R"] + '''</b></td><td class=''' + tagID(rows["R-Z score"]) + '''>''' + str(rows["R-Z score"]) + '''</td><b><td>''' + rows["L"] + '''</b></td><td class=''' + tagID(rows["L-Z score"]) + '''>''' + str(rows["L-Z score"]) + '''</td><b><td>''' + rows["Asymmetry Index (AI)"] + '''</b></td><td class=''' + tagID(rows["AI-Z score"]) + '''>''' + str(rows["AI-Z score"]) + '''</td></tr>'''
		all_entries = all_entries + '''<tbody id=''' + "section" + str(index) + '''>''' + tbody + '''</tbody>'''

	string = '''<table border=1 frame=void rules=rows>
		<thead>
			<tr style="text-align: center;">
      			<th scope="col">Brain Region</th>
      			<th scope="col">Volume (mL<sup>3</sup>)</th>
      			<th scope="col">R</th>
      			<th scope="col">R-Z score</th>
      			<th scope="col">L</th>
      			<th scope="col">L-Z score</th>
      			<th scope="col">Asymmetry Index (AI)</th>
      			<th scope="col">AI-Z score</th>
   			</tr>
  		</thead>
		''' + all_entries + '''
	</table>'''

	return string

def makeFlagTable(mydict, MUSErois):
	all_entries = ""

	# Rank by magnitude
	tmp = {}
	for key, value in mydict.items():
		tmp[key] = abs(value)

	my_keys = sorted(tmp, key=tmp.get, reverse=True)[:15]

	for index, key in enumerate(my_keys):
		tbody = ""
		lobe = list(maphemi.loc[(maphemi['ROI_NAME'] == str(key)), 'SUBGROUP_0'].values)[0]
		tbody = '''<tr style="text-align: center;"><td><b>''' + key + '''</b></td><td style="font-weight:normal">''' + str(lobe) + '''</td>''' + '''<b><td style="font-weight:normal">''' + str(round(MUSErois[key]/1000,2)) + '''</b></td><td class=''' + tagID(mydict[key]) + ''' style="font-weight:normal">''' + str(round(mydict[key],2)) + '''</td></tr>'''
		all_entries = all_entries + '''<tbody id=''' + "section" + str(index) + '''>''' + tbody + '''</tbody>'''

	string = '''<table border=1 frame=void rules=rows>
		<thead>
			<tr style="text-align: center;">
      			<th scope="col">Brain Region</th>
      			<th scope="col">Lobar Region</th>
      			<th scope="col">Volume (mL<sup>3</sup>)</th>
      			<th scope="col">Z score</th>
   			</tr>
  		</thead>
		''' + all_entries + '''
	</table>'''

	return string

#Method to get 6 axial views requested by Ilya / display these slices
def getcropinfo(img_arr):
	x_len,y_len,z_len = img_arr.shape

	y_min = 0
	y_max = 0
	slice1 = 0
	slice2 = 0
	slice3 = 0
	slice4 = 0
	slice5 = 0
	slice6 = 0

	# top to bottom slices
	### Get axial slice through bottom of occipital lobe ###
	# Get bottom of occipital lobe (higher slice value)
	ol = ((img_arr == 83) | (img_arr == 84) | (img_arr == 108) | (img_arr == 109) | (img_arr == 114) | (img_arr == 115) | (img_arr == 128) | (img_arr == 129) | (img_arr == 134) | (img_arr == 135) | (img_arr == 144) | (img_arr == 145) | (img_arr == 156) | (img_arr == 157) | (img_arr == 160) | (img_arr == 161) | (img_arr == 196) | (img_arr == 197)).astype(int)
	for y in range(1, y_len):
		if np.sum(ol[:,(y_len-y),:]) > 0:
			y_max = y_len-y
			slice6 = img_arr[:, y_max, :]
			break

	### Get axial slice through top of ventricles ###
	# Get top of ventricles (lower slice value)
	vent = ((img_arr == 4) | (img_arr == 11) | (img_arr == 49) | (img_arr == 50) | (img_arr == 51) | (img_arr == 52)).astype(int)
	for y in range(y_len):
		if np.sum(vent[:,y,:]) > 0:
			y_min = y
			slice1 = img_arr[:, y_min, :]
			break

	# Get four other slices roughly equidistant between these two defined slices;
	if (y_max - y_min) % 5 == 0:
		slice2 = img_arr[:, int(y_min + (y_max - y_min)/5), :]
		slice3 = img_arr[:, int(y_min + 2*(y_max - y_min)/5), :]
		slice4 = img_arr[:, int(y_min + 3*(y_max - y_min)/5), :]
		slice5 = img_arr[:, int(y_min + 4*(y_max - y_min)/5), :]

	else:
		start = (y_max - y_min) % 5
		slice2 = img_arr[:, int(y_min + (y_max - y_min - start)/5), :]
		slice3 = img_arr[:, int(y_min + 2*(y_max - y_min - start)/5), :]
		slice4 = img_arr[:, int(y_min + 3*(y_max - y_min - start)/5), :]
		slice5 = img_arr[:, int(y_min + 4*(y_max - y_min - start)/5), :]

	return slice1,slice2,slice3,slice4,slice5,slice6


## Create visualization of the brain structures
def generateBrainVisual(maskfile, allz, fname):
	### Check input mask
	if not maskfile:
		print("ERROR: Input file not provided!!!")
		sys.exit(0)

	### Read the input image
	roinii = nrrd.read_header(maskfile)
	roiimg = nrrd.read(maskfile)[0]

	slice1, slice2, slice3, slice4, slice5, slice6 = getcropinfo(roiimg)

	display_slicestmp = [slice1, slice2, slice3, slice4, slice5, slice6]

	### create bounding box for each slice - temp version
	for i,sli in enumerate(display_slicestmp):
		dim1,dim2 = sli.shape

		for x in range(dim1):
			if np.sum(sli[x,:]) > 0:
				x_min = x
				break

		for x in range(1, dim1):
			if np.sum(sli[(dim1-x),:]) > 0:
				x_max = dim1-x
				break

		for y in range(dim2):
			if np.sum(sli[:,y]) > 0:
				y_min = y
				break

		for y in range(1, dim2):
			if np.sum(sli[:,(dim2-y)]) > 0:
				y_max = dim2-y
				break

		display_slicestmp[i] = sli[(x_min-15):(x_max+15),(y_min-15):(y_max+15)]

	#TODO: create different arrays of same size to allow data array to preserve original values
	## We only modify the zero arrays
	my_keys = sorted(allz, key=allz.get, reverse=False)

	fig, ax = plt.subplots(2, 3, figsize=(6, 3.1), constrained_layout = True)

	for i in range(6):
		display_slicestmp[i] = np.array(display_slicestmp[i])
		mask_colors = np.zeros((display_slicestmp[i].shape[0],display_slicestmp[i].shape[1]))
		mask_grey = np.zeros((display_slicestmp[i].shape[0],display_slicestmp[i].shape[1]))

		for key in my_keys:
			if (allz[key] <= -1) and (allz[key] > -3):
				mask_colors = np.where(display_slicestmp[i] == int(key), allz[key], mask_colors)
		
		mask_grey = np.where((mask_colors != 0), 0, display_slicestmp[i])

		## Get gradient with red structures
		mask_colors = np.where((mask_grey != 0), 0, mask_colors)
		masked_array = np.ma.masked_where((mask_colors == 0), mask_colors)
		cmap1 = matplotlib.cm.Reds
		#cmap1.set_bad(color='black')

		## Get all other ROIs
		masked_array2 = np.ma.masked_where((mask_grey == 0), mask_grey)
		cmap2 = matplotlib.cm.binary

		if i == 0:
			a = 0
			b = 0
		if i == 1:
			a = 0
			b = 1
		if i == 2:
			a = 0
			b = 2
		if i == 3:
			a = 1
			b = 0
		if i == 4:
			a = 1
			b = 1
		if i == 5:
			a = 1
			b = 2

		hex_list = ['#ba1d28', '#ce352c', '#df4b33', '#ee613d', '#f67b49', '#fa9657', '#fdaf67', '#fec87a', '#fee090']
		
		pcm = ax[a,b].imshow(masked_array, cmap=get_continuous_cmap(hex_list), vmin = allz[my_keys[0]], vmax = -1, aspect='auto')
		ax[a,b].imshow(masked_array2, cmap=cmap2, vmin = 1, vmax = 300, aspect='auto')

		ax[a,b].set_xticks([])
		ax[a,b].set_yticks([])

	fig.colorbar(pcm, ax=ax[:, :], location='right').set_label('Z-score map', labelpad=15, rotation=270)

	#fig.subplots_adjust(wspace=0, hspace=0)
	#fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

	## Save to file
	#plt.tight_layout()
	plt.savefig(fname,bbox_inches='tight',pad_inches = 0.1)

################################################ END OF FUNCTIONS ################################################
	
############## MAIN ##############
#DEF
def _main(bmask, roi, icv , _json, pdf_path):

	##########################################################################
	##### Read and initial filter for reference ROI values
	dfRef = pd.read_csv(MUSE_Ref_Values).dropna()
	dfRef['Date'] = pd.to_datetime(dfRef.Date)
	dfRef = dfRef.sort_values(by='Date')
	# Get first-time points only
	dfRef = dfRef.drop_duplicates(subset=['PTID'], keep='first')
	# Binarize sex
	dfRef['Sex'].replace(0, 'F',inplace=True)
	dfRef['Sex'].replace(1, 'M',inplace=True)

	##########################################################################
	##### Read subject data (demog and MRI)

	####################################
	## Read subject/scan info
	subDict = readSubjInfo(_json)
	
	## Create subject dataframe with all input values
	#dfSub = pd.DataFrame(dfRef.columns)
	dfSub = pd.DataFrame(columns=['MRID','Age','Sex'])
	dfPat = pd.DataFrame(columns=[''])
	dfSub.loc[0,'MRID'] = str(subDict['MRID'])
	dfSub.loc[0,'Age'] = float(subDict['Age'])
	dfSub.loc[0,'Sex'] = str(subDict['Sex'])

	dfPat.loc[0] = str(subDict['MRID'])
	dfPat.loc[1] = str(subDict['Age'])
	dfPat.loc[2] = str(subDict['Sex'])
	dfPat.loc[3] = str(subDict['ExamDate'])
	dfPat.loc[4] = datetime.today().strftime('%m-%d-%Y')

	##########################################################################
	##### Path where we'll save the brain structure images to be displayed
	tmpOut_brainplt = '/tmpfolder/' + _os.path.basename(pdf_path.removesuffix(".pdf") + '_brainplot.png')

	## Read MRI values
	## Read bmask, if provided
	#bmaskVol = None
	#if len(bmask) == 1:
	#	generateBrainVisual(bmask[0], tmpOut_brainplt)
	#	bmaskVol = calcMaskVolume(bmask[0])
	#	#print('bmask : ' + str(bmaskVol))
	#	dfSub['MUSE_BrainMask'] = bmaskVol

	##########################################################################
	##### Path where we'll save the brain structure images to be displayed
	## Read icv, if provided
	icvVol = None
	dfSub['ICV'] = None
	if len(icv) == 1:
		icvVol = calcMaskVolume(icv[0])
		dfSub['ICV'] = icvVol

	####################################
	## Read MUSE file, if provided; outputs - dictionary of gender and age adjusted z-scores (allz); 
	## dictionary of all MUSE ROIs (all_MuseROIs); dictionary of ICV-adjusted reference values in cm^3 (dfRef)
	## 
	roiVols = None
	allz = None
	spareAD = None
	spareBA = None

	if len(roi) == 1:
		roiVols, dfRef, allz, allz_num, all_MuseROIs_num, all_MuseROIs_name = calcRoiVolumes(roi[0], MUSE_ROI_Mapping, dfRef, subDict)
		generateBrainVisual(roi[0], allz_num, tmpOut_brainplt)
		spareAD, prediction = calculateSpareAD(subDict['Age'],subDict['Sex'],all_MuseROIs_num)
		spareBA = calculateSpareBA(subDict['Age'],subDict['Sex'],all_MuseROIs_num)

		for key in roiVols.keys():
			dfSub.loc[0, key] = roiVols[key]

	##########################################################################
	##########################################################################
	dfRef.rename(columns={'702':'ICV'}, inplace=True)
	##### TO DO (03/22/2022): ICV correct subject MUSE values (only if ICV mask is provided)
	## ICV-adjustment for subject values
	nonROI = list(dfSub.columns[dfSub.columns.str.contains('AI')].values)
	nonROI.extend(['MRID','Age','Sex','ICV'])


	if dfSub['ICV'] is not None:
		dfSub[dfSub.columns.difference(nonROI)] = dfSub[dfSub.columns.difference(nonROI)].div(dfSub['ICV'], axis=0)*dfRef['ICV'].mean()

	# Convert from mm^3 to cm^3
	dfSub[dfSub.columns.difference(nonROI)] = dfSub[dfSub.columns.difference(nonROI)]/1000

	#### ICV-adjustment for reference values ####
	nonROI = list(dfRef.columns[dfRef.columns.str.contains('AI')].values)
	nonROI.extend(['MRID','Study','PTID','Age','Sex','Diagnosis_nearest_2.0','SITE','Date','ICV','SPARE_AD','SPARE_BA'])

	# Select subjects with the same sex
	dfRef = dfRef[dfRef.Sex == subDict['Sex']]

	# Make copy for reference value plotting
	dfReftmp = dfRef.copy(deep=True)

	dfRef['Age'] = dfRef['Age'].round(0)
	dfRef = dfRef[dfRef.Age == round(subDict['Age'],0)]

	# Actually do the ICV-adjustment
	dfRef[dfRef.columns.difference(nonROI)] = dfRef[dfRef.columns.difference(nonROI)].div(dfRef['ICV'], axis=0)*dfRef['ICV'].mean()

	# TODO: Should I be ICV-adjusting for only +- 5 years?
	dfReftmp[dfReftmp.columns.difference(nonROI)] = dfReftmp[dfReftmp.columns.difference(nonROI)].div(dfReftmp['ICV'], axis=0)*dfReftmp['ICV'].mean()

	# Convert from mm^3 to cm^3
	dfRef[dfRef.columns.difference(nonROI)] = dfRef[dfRef.columns.difference(nonROI)]/1000
	dfReftmp[dfReftmp.columns.difference(nonROI)] = dfReftmp[dfReftmp.columns.difference(nonROI)]/1000

	##########################################################################
	##### Create plots and tables
	plots = []
	brainplot = []
	imgs = []
	tables = []
	trial_set = []

	## Setup dataframe to store numeric details on ROIs
	tmpTable = pd.DataFrame(columns=["Brain Region", "Volume (cubic mL)", "R", "R-Z score", "L", "L-Z score", "Asymmetry Index (AI)", "AI-Z score"])#, "Total normative percentile"])

	### Get chosen roi volumes into this table
	names = [i.lstrip("Total ") for i in roiVols.keys() if "Total" in i]

	if len(roi) == 1:
		trial_set = [i for i in list(roiVols.keys()) if "Total" in i]
		for key in names:
			#totalnorm = round(stats.percentileofscore(dfRef[dfRef.Age == subDict['Age']]["Total " + key].values, dfSub["Total " + key].values[0], kind='rank'))
			# Check if roi has L/R component
			if sum(key in i for i in roiVols.keys()) == 1:
				tmpTable.loc[len(tmpTable.index)] = [key, str(round(dfSub["Total " + key].values[0],2)), "-","-","-","-","-","-"]#, totalnorm]
			# If roi does have L/R component
			else:
				Rnorm = round((dfSub["Right " + key].values[0] - dfRef["Right " + key].mean())/dfRef["Right " + key].std(),2)
				Lnorm = round((dfSub["Left " + key].values[0] - dfRef["Left " + key].mean())/dfRef["Left " + key].std(),2)
				## TODO: verify this is accurate way to z-score AI index ##
				AInorm = round((dfSub[key + " AI"].abs().values[0] - dfRef[key + " AI"].abs().mean())/dfRef[key + " AI"].abs().std(),2)
				tmpTable.loc[len(tmpTable.index)] = [key, str(round(dfSub["Total " + key].values[0],2)), str(round(dfSub["Right " + key].values[0],2)), Rnorm, str(round(dfSub["Left " + key].values[0],2)), Lnorm, str(round(dfSub[key + " AI"].values[0],2)), AInorm]#, totalnorm]

	### Get icv volume into this table
	#if len(icv) == 1:
	#	#trial_set.append('ICV')
	#	totalnorm = round((dfSub["ICV"].values[0] - dfRef["ICV"].mean())/dfRef["ICV"].std(),2)
	#	tmpTable.loc[len(tmpTable.index)] = ["ICV", str(int(dfSub["ICV"].values[0])), "-","-","-","-","-","-"]#, totalnorm]

	### Get bmask volume into this table
	if len(bmask) == 1:
		trial_set.append('MUSE_BrainMask')
		totalnorm = round((dfSub["MUSE_BrainMask"].values[0] - dfRef["701"].mean())/dfRef["701"].std(),2)
		tmpTable.loc[len(tmpTable.index)] = ["MUSE_BrainMask", str(round(dfSub["MUSE_BrainMask"].values[0],2)), "-","-","-","-","-","-"]#, totalnorm]

	tmpOut = '/tmpfolder/' + _os.path.basename(pdf_path.removesuffix(".pdf") + '_plot.png')

	plotWithRef(dfReftmp, dfSub, trial_set, tmpOut, spareAD, spareBA)

	plots.append(_os.path.basename(tmpOut))
	brainplot.append(_os.path.basename(tmpOut_brainplt))
	imgs.append(tmpOut)
	imgs.append(tmpOut_brainplt)

	## Create brain volume table and flag table (get the HTML)
	tables = [makeTables(tmpTable)]
	flagtable = [makeFlagTable(allz, all_MuseROIs_name)]

	## Generate final html and then convert to pdf file
	html_out = '/tmpfolder/' + _os.path.basename(pdf_path.removesuffix("pdf") + "html")
	writeHtml(plots, tables, flagtable, dfPat, html_out, brainplot)
	print('\nTemp html file created: ' + html_out)
	wp(html_out).write_pdf(pdf_path)
	print('\nPDF plot file created: ' + pdf_path)

	## Remove temp html files
	_os.remove(html_out)

	## Remove all temp plot images
	for i in imgs:
		_os.remove(i)

if __name__ == '__main__':
	_main(nrrd,json,pdf_path)
