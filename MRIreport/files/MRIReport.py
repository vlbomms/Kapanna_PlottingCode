import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

from scipy import stats

from pygam import ExpectileGAM
from pygam.datasets import mcycle
from pygam import LinearGAM

from IPython.display import HTML as idhtml
from weasyprint import HTML as wp
### Constants (TODO: temporary)
MUSE_ROI_Mapping = '~/Ref_CSV/MUSE_DerivedROIs_Mappings.csv'
MUSE_Ref_Values = '~/Ref_CSV/RefStudy_MRVals.csv'
MUSE_ExtendedRef_Values = '~/Ref_CSV/RefStudy_MRVals_AllVals.csv'
SEL_ROI = ['MUSE_Volume_702', 'MUSE_Volume_701', 'MUSE_Volume_601', 'MUSE_Volume_604', 'MUSE_Volume_509', 'MUSE_Volume_48', 'MUSE_Volume_47']
SEL_ROI_Rename = ['MUSE_BrainMask', 'MUSE_TotalBrain', 'MUSE_GM', 'MUSE_WM', 'MUSE_VN', 'MUSE_HippoL', 'MUSE_HippoR']

################################################ FUNCTIONS ################################################

############## HELP #############

### Function to read demog. info from json
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
			subAge = float([value for key, value in data.items() if 'PatientAge' in key][0].replace('Y',''))
			subSex = [value for key, value in data.items() if 'PatientSex' in key][0]
			subExamDate = [value for key, value in data.items() if 'StudyDate_date' in key][0]
			
			subDict = {'MRID':subID, 'Age':subAge, 'Sex':subSex, "ExamDate":subExamDate}

	return subDict


### Function to calculate mask volume
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
def calcRoiVolumes(maskfile, mapcsv):
	
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
			# Append the ROI number to the list
			DerivedROIs.append(row[0])
			roiInds = [int(x) for x in row[2:]]
			DerivedVols.append(np.sum(VolumesInd[roiInds]))

	### Decide what to output
	all_MuseROIs =  dict(zip(DerivedROIs, DerivedVols))
	dictr = {}
	
	### Extract ONLY the ROI we're interested in -- based on SEL_ROI
	for roi in SEL_ROI:
		x = [i for i in roi.split('_') if i.isdigit()][0]
		dictr[x] = all_MuseROIs[x]

	return dictr, all_MuseROIs

### Rewrite to get coplots!!!
def plotWithRef(dfRef, dfSub, selVarlst, fname):
	fig = py.subplots.make_subplots(rows=round(len(selVarlst)/2),cols=2, subplot_titles=selVarlst)
	row = 1
	column = 1
	sl = False

	for selVar in selVarlst:
		## Only allow legend to show up for one of the plots (for display purpose)
		if row == 1 & column == 1:
			sl = True
		else:
			sl = False

		X = dfRef.Age.values.reshape([-1,1])
		y = dfRef[selVar].values.reshape([-1,1])

		##############################################################
		## Fit expectiles
		# fit the mean model first by CV
		gam50 = ExpectileGAM(expectile=0.5).gridsearch(X, y)

		# and copy the smoothing to the other models
		lam = gam50.lam

		# fit a few more models
		gam95 = ExpectileGAM(expectile=0.95, lam=lam).fit(X, y)
		gam75 = ExpectileGAM(expectile=0.75, lam=lam).fit(X, y)
		gam25 = ExpectileGAM(expectile=0.25, lam=lam).fit(X, y)
		gam05 = ExpectileGAM(expectile=0.05, lam=lam).fit(X, y)
	
		XX = gam50.generate_X_grid(term=0, n=100)
		XX95 = list(gam95.predict(XX).flatten())
		XX75 = list(gam75.predict(XX).flatten())
		XX50 = list(gam50.predict(XX).flatten())
		XX25 = list(gam25.predict(XX).flatten())
		XX05 = list(gam05.predict(XX).flatten())
		XX = list(XX.flatten())

		fig.append_trace( go.Scatter(mode='markers', x=dfRef.Age.tolist(), y=dfRef[selVar].tolist(), marker=dict(color='MediumBlue', size=10), opacity=0.2, showlegend=False),row,column)
		fig.append_trace( go.Scatter(mode='lines', x=XX, y=XX95, marker=dict(color='MediumPurple', size=10), name='95th percentile', showlegend=sl),row,column)
		fig.append_trace( go.Scatter(mode='lines', x=XX, y=XX75, marker=dict(color='Orchid', size=10), name='75th percentile', showlegend=sl),row,column)
		fig.append_trace( go.Scatter(mode='lines', x=XX, y=XX50, marker=dict(color='MediumVioletRed', size=10), name='50th percentile', showlegend=sl),row,column)
		fig.append_trace( go.Scatter(mode='lines', x=XX, y=XX25, marker=dict(color='Orchid', size=10), name='25th percentile', showlegend=sl),row,column)
		fig.append_trace( go.Scatter(mode='lines', x=XX, y=XX05, marker=dict(color='MediumPurple', size=10), name='5th percentile', showlegend=sl),row,column)
		fig.append_trace( go.Scatter(mode='markers', x=dfSub.Age.tolist(), y=dfSub[selVar].tolist(),marker=dict(color='Red', size=16,line=dict( color='MediumPurple', width=2)), name='Patient', showlegend=sl),row,column)

		## Allow iteration through nx2 structure of plots
		if row == 1 & column == 1:
			column += 1
		elif column == 1:
			column += 1
		else:
			row += 1
			column -= 1

	## Get Age underneath each plot
	for i in range(1,len(selVarlst)+1):
		fig['layout']['xaxis{}'.format(i)]['title']='Age'

	## Set figure dimensions and save
	fig.update_layout(margin_b=0,margin_l=0,margin_r=0,margin_t=17)
	fig.write_image(fname, scale=1, width=550, height=550)
	

## Write the HTML code for display --- TODO: Move to seperate python file
def writeHtml(plots, brainplot, tables, pattable, outName):
	all_plot = ""
	all_brainplot = ""
	all_table = ""

	#Generating HTML for plots and tables
	for plot in plots:
		ind = '''<img src=''' + plot + '''></img>'''
		all_plot = all_plot + ind

	for table in tables:
		ind = '' + table + ''
		all_table = all_table + ind
	for plot in brainplot:
		ind = '''<img src=''' + plot + '''></img>'''
		all_brainplot = all_brainplot + ind
	#HTML code for report structure
	html_string_test = '''
	<html>
	<head>
		<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
		<style>
			body { margin:0 100; #background:whitesmoke; }
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
			.grouplabel {
				background: blue;
				color: yellow;
				border: 1px solid blue;
				border-radius: 5px;
			}
		</style>
		
	</head>
	<body>
		<div class="container">
			<h1>Report</h1>
		</div>
		<div class="container">
			''' + pattable + '''
		</div>
		<div class="container">
			<h6>Brain structures: </h6>
			''' + all_brainplot + '''
		</div>		
		<div class="container">
			<h6>Age plots: </h6>
			''' + all_plot + '''
		</div>
		<div class="container">
			<h6>Brain structure volumes: </h6>
			''' + all_table + '''
		</div>
	</body>
	</html>'''

	f = open(outName,'w')
	f.write(html_string_test)
	f.close()

def tagID(num):
	if num > 95:
		tag = "toohigh"
	elif num <= 40:
		tag = "toolow"
	else:
		tag = "norm"

	return tag

## Create table with volume values + percentiles; TODO: add mechanic for subtables
def makeTables(dftable, subtables, superstruct_ID):
	all_entries = ""
	
	for index, rows in dftable.iterrows():
		tbody = ""
		# Access dictionary with subTable information!!!
		try:
			tbody = '''<tr style="text-align: center;"><td><b><u>''' + rows["Brain Structure"] + '''</u></b></th><td>''' + rows["Volume"] + '''</td><td class=''' + tagID(rows["Normative Percentile"]) + '''>''' + str(rows["Normative Percentile"]) + '''</td></tr>'''
			for index_sub, rows_sub in subtables[superstruct_ID[rows["Brain Structure"]]].iterrows():
				tbody = tbody + '''<tr style="text-align: center;">
				<td scope="col"><i>''' + str(rows_sub["Brain Structure"]) + '''</i></td>
				<td scope="col"><i>''' + str(rows_sub["Volume"]) + '''</i></td>
				<td class=''' + tagID(rows_sub["Normative Percentile"]) + '''><i>''' + str(rows_sub["Normative Percentile"]) + '''</i></td>
				</tr>'''

			all_entries = all_entries + '''<tbody id=''' + "section" + str(index) + '''; class="grouplabel">''' + tbody + '''</tbody>'''

		except:
			tbody = ""
			tbody = tbody + '''<tr style="text-align: center;">
				<td scope="col">''' + str(rows["Brain Structure"]) + '''</td>
				<td scope="col">''' + str(rows["Volume"]) + '''</td>
				<td class=''' + tagID(rows["Normative Percentile"]) + '''>''' + str(rows["Normative Percentile"]) + '''</td>
				</tr>'''

			all_entries = all_entries + '''<tbody id=''' + "section" + str(index) + '''>''' + tbody + '''</tbody>'''

	string = '''<table>
  		<thead>
    		<tr style="text-align: center;">
      			<th scope="col">Brain Structure</th>
      			<th scope="col">Volume</th>
      			<th scope="col">Normative Percentile</th>
   			</tr>
  		</thead>
		''' + all_entries + '''
	</table>'''

	###TODO: Change above code to accomodate sub-DataFrames containing substructures; https://stackoverflow.com/a/49458729 <<- example on how to do so
	return string

## Make patient, site, and report information tables
def makeIntroTables(dftable):
	string = '''
	<div class="col-xs-4">
		<h6 class="sub-header">Patient information</h6>
			<b>Patient ID: </b>''' + dftable.loc[0].values[0] + '''
				<br>
			<b>Age: </b>''' + dftable.loc[1].values[0] + '''
				<br>
			<b>Sex: </b>''' + dftable.loc[2].values[0] + '''
				<br>
	</div>
	<div class="col-xs-4">
		<h6 class="sub-header">Report information</h6>
			<b>Scan Date: </b>''' + dftable.loc[3].values[0] + '''
				<br>
			<b>Report Date: </b>''' + dftable.loc[4].values[0] + '''
				<br>
	</div>
	<div class="col-xs-4">
		<h6 class="sub-header">Site information</h6>
			<b>Perelman Medical School</b>
				<br>
	</div>
	'''

	return string


## Get center of brain mask for dislay purposes; TODO: Move to seperate python file
def getcropinfo(img_arr):
	## finding a 3D bounding box
	x_len,y_len,z_len = img_arr.shape

	for x in range(x_len):
		if np.sum(img_arr[x,:,:]) > 0:
			x_min = x
			break

	for x in range(1, x_len):
		if np.sum(img_arr[(x_len-x),:,:]) > 0:
			x_max = x_len-x
			break

	for y in range(y_len):
		if np.sum(img_arr[:,y,:]) > 0:
			y_min = y
			break

	for y in range(1, y_len):
		if np.sum(img_arr[:,(y_len-y),:]) > 0:
			y_max = y_len-y
			break

	for z in range(z_len):
		if np.sum(img_arr[:,:,z]) > 0:
			z_min = z
			break

	for z in range(1, z_len):
		if np.sum(img_arr[:,:,(z_len-z)]) > 0:
			z_max = z_len-z
			break

	x_center = int(np.mean((x_min, x_max)))
	y_center = int(np.mean((y_min, y_max)))
	z_center = int(np.mean((z_min, z_max)))

	return x_center, y_center, z_center


## Create visualization of the brain structures
def generateBrainVisual(maskfile, fname):
	### Check input mask
	if not maskfile:
		print("ERROR: Input file not provided!!!")
		sys.exit(0)

	### Read the input image
	roinii = nrrd.read_header(maskfile)
	roiimg = nrrd.read(maskfile)[0]
	#roiimg = np.rot90(roiimg)

	x_center,y_center,z_center = getcropinfo(roiimg)
	x_slice = np.rot90(roiimg[x_center, :, :])
	y_slice = np.rot90(roiimg[:, y_center, :])
	z_slice = np.rot90(roiimg[:, :, z_center])

	display_slices = [x_slice,y_slice,z_slice]

	fig, ax = plt.subplots(1, 3, figsize=[18, 3])

	##Color and display settings
	for i in range(3):
		ax[i].imshow(display_slices[i], 'RdBu_r')
		ax[i].set_xticks([])
		ax[i].set_yticks([])

	fig.subplots_adjust(wspace=0, hspace=0)

	## Save to file
	#figure.set_size_inches(6, 6)
	plt.savefig(fname)

## TODO: Get out of using this redundant function
def indextoname(num):
	index = [idx for idx, s in enumerate(SEL_ROI) if str(num) in s][0]

	return SEL_ROI_Rename[index]



################################################ END OF FUNCTIONS ################################################
	
############## MAIN ##############
#DEF
def _main(bmask, roi, icv , _json, pdf_path):

	##########################################################################
	##### Read reference ROI values
	dfRef = pd.read_csv(MUSE_Ref_Values).dropna()
	dfRefExtended = pd.read_csv(MUSE_ExtendedRef_Values).dropna()

	##########################################################################
	##### Read subject data (demog and MRI)

	####################################3
	## Read subject/scan info
	subDict = readSubjInfo(_json)

	## Create patient dataframe with all input values
	dfSub = pd.DataFrame(columns=dfRef.columns)

	## Create patient demographic info table
	dfPat = pd.DataFrame(columns=[''])
	dfSub.loc[0,'MRID'] = str(subDict['MRID'])
	dfSub.loc[0,'Age'] = int(subDict['Age'])
	dfSub.loc[0,'Sex'] = str(subDict['Sex'])

	dfPat.loc[0] = str(subDict['MRID'])
	dfPat.loc[1] = str(subDict['Age'])
	dfPat.loc[2] = str(subDict['Sex'])
	dfPat.loc[3] = str(subDict['ExamDate'])
	dfPat.loc[4] = datetime.today().strftime('%Y-%m-%d')

	####################################3

	## Path where we'll save the brain structure images to be displayed
	tmpOut_brainplt = '~/tmpfolder/' + _os.path.basename(pdf_path.removesuffix(".pdf") + '_brainplot.png')	

	## Read bmask, if provided
	bmaskVol = None
	if len(bmask) == 1:
		generateBrainVisual(bmask[0], tmpOut_brainplt)
		bmaskVol = calcMaskVolume(bmask[0])
	dfSub.MUSE_Volume_702 = bmaskVol

	## Read icv, if provided
	icvVol = None
	if len(icv) == 1:
		generateBrainVisual(icv[0], tmpOut_brainplt)
		icvVol = calcMaskVolume(icv[0])
	dfSub.DLICV = icvVol

	## Read roi, if provided
	roiVols = None
	if len(roi) == 1:
		generateBrainVisual(roi[0], tmpOut_brainplt)
		roiVols, FullMuseDict = calcRoiVolumes(roi[0], MUSE_ROI_Mapping)
		for tmpRoi in SEL_ROI:
			if tmpRoi.replace('MUSE_Volume_','') in roiVols:
				dfSub.loc[0, tmpRoi] = roiVols[tmpRoi.replace('MUSE_Volume_','')]

	##########################################################################
	##### Rename ROIs
	dictMUSE = dict(zip(SEL_ROI,SEL_ROI_Rename))
	dfRef = dfRef.rename(columns=dictMUSE)
	dfSub = dfSub.rename(columns=dictMUSE)

	# Make column for WM + GM; as well as any future summed values
	dfSub["MUSE_GM&WM"] = pd.to_numeric(dfSub["MUSE_WM"]) + pd.to_numeric(dfSub["MUSE_GM"])
	dfRef["MUSE_GM&WM"] = pd.to_numeric(dfRef["MUSE_WM"]) + pd.to_numeric(dfRef["MUSE_GM"])

	dfSub = dfSub.replace(0,np.nan).dropna(axis=1,how="all")

	# ##########################################################################
	# ##### ICV correct MUSE values
	
	# ## Correct ref values (Doesn't seem to work/have purpose as of now - 9/28/2021 - functionality unclear)
	# dfRefTmp = dfRef[dfRef.columns[dfRef.columns.str.contains('MUSE_')]]
	# dfRefTmp = dfRefTmp.div(dfRef.DLICV, axis=0)*dfRef.DLICV.mean()
	# dfRefTmp = dfRefTmp.add_suffix('_ICVCorr')
	# dfRef = pd.concat([dfRef, dfRefTmp], axis=1)

	# ## Correct sub values
	# dfSubTmp = dfSub[dfSub.columns[dfSub.columns.str.contains('MUSE_')]]
	# dfSubTmp = dfSubTmp.div(dfSub.DLICV, axis=0)*dfRef.DLICV.mean()
	# dfSubTmp = dfSubTmp.add_suffix('_ICVCorr')
	# dfSub = pd.concat([dfSub, dfSubTmp], axis=1)

	### Create an extended version of reference values
	## Select only subjects with the same sex
	dfRef = dfRef[dfRef.Sex == subDict['Sex']]
	dfRefExtended["ID"] = dfRefExtended["ID"].astype(str).str[4:].astype(int)
	all_subj = list(set(dfRefExtended.ID.values).intersection(dfRef.MRID.values))
	###Create the extended reference value dataframe
	dfRefExtended = dfRefExtended[dfRefExtended["ID"].isin(all_subj)]
	dfRef = dfRef[dfRef["MRID"].isin(all_subj)]
	dfRefExtended["Age"] = dfRef["Age"].values

	##########################################################################
	
	##### Create plots and tables

	plots = []
	brainplot = []
	imgs = []
	tables = []
	trial_set = []
	subtables = {}
	substruct_name = {}
	superstruct_ID = {}
	
	## Setup dataframe to store numeric details on ROIs; need to run tmpTable for each of the substructures (one table for each overarching)
	tmpTable = pd.DataFrame(columns=["Brain Structure", "Volume", "Normative Percentile"])

	## purpose: build reference dict for substructure names
	with open(MUSE_ROI_Mapping) as mapcsvfile:
		reader = _csv.reader(mapcsvfile, delimiter=',')
		for row in reader:
			substruct_name[row[0]] = row[1]


	## Create dict of substructure tables
	for i in roiVols:
		if i == "702" or i == "701" or i == "601" or i == "604":
			continue
		with open(MUSE_ROI_Mapping) as mapcsvfile:
			reader = _csv.reader(mapcsvfile, delimiter=',')
			# Read each line in the csv map files
			for ind, row in enumerate(reader):
				if row[0] == i and len(row[2:]) > 1:
					tmp_struct_table = pd.DataFrame(columns=["Brain Structure", "Volume", "Normative Percentile"])
					for j in row[2:]:
						tmp_struct_table.loc[len(tmp_struct_table.index)] = [substruct_name[j], str(int(FullMuseDict[j])), round(stats.percentileofscore(dfRefExtended[dfRefExtended.Age == subDict['Age']][j].values, FullMuseDict[j], kind='rank'))]

					subtables[row[0]] = tmp_struct_table
					## Have to do this b/c zipping conversion earlier btwn name and #
					superstruct_ID[indextoname(row[0])] = row[0]

	### Get icv volume into a table
	if len(icv) == 1:
		trial_set.append('DLICV')
		tmpTable.loc[len(tmpTable.index)] = ['DLICV', str(int(dfSub['DLICV'].values[0])), round(stats.percentileofscore(dfRef[dfRef.Age == subDict['Age']]['DLICV'].values, dfSub['DLICV'].values[0], kind='rank'))]

	### Get bmask volume into a table
	if len(bmask) == 1:
		trial_set.append('MUSE_BrainMask')
		tmpTable.loc[len(tmpTable.index)] = ['MUSE_BrainMask', str(int(dfSub['MUSE_BrainMask'].values[0])), round(stats.percentileofscore(dfRef[dfRef.Age == subDict['Age']]['MUSE_BrainMask'].values, dfSub['MUSE_BrainMask'].values[0], kind='rank'))]

	## Extract all important ROI labels into a list
	ind = list(dfSub.columns).index("Sex")
	trial_set = trial_set + list(dfSub.columns[(ind+1):])

	## Get roi volume into a table
	if len(roi) == 1:
		for i in trial_set:
			tmpTable.loc[len(tmpTable.index)] = [i, str(int(dfSub[i].values[0])), round(stats.percentileofscore(dfRef[dfRef.Age == subDict['Age']][i].values, dfSub[i].values[0], kind='rank'))]

	tmpOut = '~/tmpfolder/' + _os.path.basename(pdf_path.removesuffix(".pdf") + '_plot.png')
	plotWithRef(dfRef, dfSub, trial_set, tmpOut)
	plots.append(_os.path.basename(tmpOut))
	brainplot.append(_os.path.basename(tmpOut_brainplt))
	imgs.append(tmpOut)
	imgs.append(tmpOut_brainplt)

	## Create brain volume table (in HTML)
	tables = [makeTables(tmpTable, subtables, superstruct_ID)]

	### Create patient overall biodata table (in HTML)
	tmpPatTable = makeIntroTables(dfPat)

	html_out = '~/tmpfolder/' + _os.path.basename(pdf_path.removesuffix("pdf") + "html")
	writeHtml(plots, brainplot, tables, tmpPatTable, html_out)
	print('\nTemp html file created: ' + html_out)
	## Convert html to pdf file
	wp(html_out).write_pdf(pdf_path)
	print('\nPDF plot file created: ' + pdf_path)

	## Remove temp html files
	_os.remove(html_out)

	## Remove all temp plot images
	for i in imgs:
		_os.remove(i)

if __name__ == '__main__':
	_main(nrrd,json,pdf_path)