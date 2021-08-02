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

from scipy import stats

from pygam import ExpectileGAM
from pygam.datasets import mcycle
from pygam import LinearGAM

#import plotly as py
#import plotly.plotly as py
#import plotly.tools as plotly_tools
#import plotly.graph_objs as go

from IPython.display import HTML as idhtml
from weasyprint import HTML as wp


### Constants (TODO: temporary)
MUSE_ROI_Mapping = '/Ref_CSV/MUSE_DerivedROIs_Mappings.csv'
MUSE_Ref_Values = '/Ref_CSV/RefStudy_MRVals.csv'
SEL_ROI = ['MUSE_Volume_702', 'MUSE_Volume_701', 'MUSE_Volume_601', 'MUSE_Volume_604', 'MUSE_Volume_509', 'MUSE_Volume_48']
SEL_ROI_Rename = ['MUSE_BrainMask', 'MUSE_TotalBrain', 'MUSE_GM', 'MUSE_WM', 'MUSE_VN', 'MUSE_HippoL']

################################################ FUNCTIONS ################################################

############## HELP #############

### Function to read demog info from json
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
	return dict(zip(DerivedROIs, DerivedVols))

def plotWithRef(dfRef, dfSub, selVar, fname):

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

	fig = px.scatter(dfRef, x='Age', y=selVar, opacity=0.2)
	fig.add_trace( go.Scatter(mode='lines', x=XX, y=XX95, marker=dict(color='MediumPurple', size=10), name='perc95'))
	fig.add_trace( go.Scatter(mode='lines', x=XX, y=XX75, marker=dict(color='Orchid', size=10), name='perc75'))
	fig.add_trace( go.Scatter(mode='lines', x=XX, y=XX50, marker=dict(color='MediumVioletRed', size=10), name='perc50'))
	fig.add_trace( go.Scatter(mode='lines', x=XX, y=XX25, marker=dict(color='Orchid', size=10), name='perc25'))
	fig.add_trace( go.Scatter(mode='lines', x=XX, y=XX05, marker=dict(color='MediumPurple', size=10), name='perc05'))
	fig.add_trace( go.Scatter(
		mode='markers', x=dfSub.Age.tolist(), y=dfSub[selVar].tolist(),marker=dict(color='Red', size=16,line=dict( color='MediumPurple', width=2)), name='Sub'))
	
	#fig.write_html(fname, include_plotlyjs = 'cdn')
	fig.write_image(fname)#, include_plotlyjs = 'cdn')
	
def plotToHtml(dfRef, dfSub, selVar, fname):

	x = dfRef.Age.values.tolist()
	y = dfRef[selVar].values.tolist()
	
	fig = px.scatter(dfRef, x='Age', y=selVar)
	fig.add_trace( go.Scatter(
		mode='markers', x=dfSub.Age.tolist(), y=dfSub[selVar].tolist(),marker=dict(color='Red', size=16,line=dict( color='MediumPurple', width=2))))
	#fig = px.scatter(x=dfSub.Age.tolist(), y=dfSub[selVar].tolist(), color=['Red'])
	
	#px.scatter(x=dfSub.Age.tolist(), y=dfSub[selVar].tolist(), color="species")
	
	fig.write_html(fname)#, include_plotlyjs = 'cdn')

	#plt.plot(dfSub.Age, dfSub[selVar], 'ro', markersize=10)
	#plt.xlabel('Age')
	#plt.ylabel(selVar)
	

def writeHtml(plots, tables, pattable, outName):

	#penn_logo = "/logos/UniversityofPennsylvania_FullLogo_RGB_0.png"
	#pennmed_logo = "/logos/PennMedicineLogo.png"
	#cbica_logo = "/logos/cbica.png"
	comb_logo = "/logos/comb.png"

	html_string_test = '''
	<html>
	<head>
		<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
		<style>body{ margin:0 100; background:whitesmoke; }</style>
	</head>
	<body>
		<img src="''' + comb_logo + '''" width="100" height="100" /></img>

		<h1>Report automatically generated by CBICA Deep Learning Neuro Workbench (aka Kaapana platform)</h1>

		<h3>Patient and Study Information:</h3>
		''' + pattable + '''

		<h3>Age plot:</h3>
		<img width="1000" height="550" \
	src="''' + plots[0] + '''"></img>

		<h3>Reference table: </h3>
		''' + tables[0] + '''

		<h3>Findings: </h3>

		<h3>Recommendations: </h3>

		<h3>Notes: </h3>

	</body>
	</html>'''

	f = open(outName,'w')
	#f.write(html_string)
	f.write(html_string_test)
	f.close()

################################################ END OF FUNCTIONS ################################################
	
############## MAIN ##############
#DEF
def _main(bmask, roi, icv , _json, pdf_path):

	##########################################################################
	##### Read reference ROI values
	dfRef = pd.read_csv(MUSE_Ref_Values).dropna()

	##########################################################################
	##### Read subject data (demog and MRI)

	####################################3
	## Read subject/scan info
	subDict = readSubjInfo(_json)
	
	## Create subject dataframe with all input values
	dfSub = pd.DataFrame(columns=dfRef.columns)
	dfPat = pd.DataFrame(columns=['Patient Name','Age','Gender','Exam Date','Report Created Date'])
	dfSub.loc[0,'MRID'] = str(subDict['MRID'])
	dfSub.loc[0,'Age'] = float(subDict['Age'])
	dfSub.loc[0,'Sex'] = str(subDict['Sex'])

	dfPat.loc[0,'Patient Name'] = str(subDict['MRID'])
	dfPat.loc[0,'Age'] = float(subDict['Age'])
	dfPat.loc[0,'Gender'] = str(subDict['Sex'])
	dfPat.loc[0,'Exam Date'] = str(subDict['ExamDate'])
	dfPat.loc[0,'Report Created Date'] = datetime.today().strftime('%Y-%m-%d')

	####################################3
	## Read MRI values

	## Read bmask, if provided
	bmaskVol = None
	if len(bmask) == 1:
		bmaskVol = calcMaskVolume(bmask[0])
		#print('bmask : ' + str(bmaskVol))
	dfSub.MUSE_Volume_702 = bmaskVol

	## Read icv, if provided
	icvVol = None
	if len(icv) == 1:
		icvVol = calcMaskVolume(icv[0])
		#print('icv : ' + str(icvVol))
	dfSub.DLICV = icvVol

	## Read roi, if provided
	roiVols = None
	if len(roi) == 1:
		roiVols = calcRoiVolumes(roi[0], MUSE_ROI_Mapping)
		for tmpRoi in SEL_ROI:
			if tmpRoi.replace('MUSE_Volume_','') in roiVols:
				dfSub.loc[0, tmpRoi] = roiVols[tmpRoi.replace('MUSE_Volume_','')]
		
	##########################################################################
	##### Rename ROIs
	dictMUSE = dict(zip(SEL_ROI,SEL_ROI_Rename))
	dfRef = dfRef.rename(columns=dictMUSE)
	dfSub = dfSub.rename(columns=dictMUSE)
	#print(dfSub.columns)

	##########################################################################
	##### ICV correct MUSE values
	
	## Correct ref values
	dfRefTmp = dfRef[dfRef.columns[dfRef.columns.str.contains('MUSE_')]]
	dfRefTmp = dfRefTmp.div(dfRef.DLICV, axis=0)*dfRef.DLICV.mean()
	dfRefTmp = dfRefTmp.add_suffix('_ICVCorr')
	dfRef = pd.concat([dfRef, dfRefTmp], axis=1)

	## Correct sub values
	dfSubTmp = dfSub[dfSub.columns[dfSub.columns.str.contains('MUSE_')]]
	dfSubTmp = dfSubTmp.div(dfSub.DLICV, axis=0)*dfRef.DLICV.mean()
	dfSubTmp = dfSubTmp.add_suffix('_ICVCorr')
	dfSub = pd.concat([dfSub, dfSubTmp], axis=1)


	## Select subjects with the same sex
	dfRef = dfRef[dfRef.Sex == subDict['Sex']]

	##########################################################################
	##### Create plots and tables
	##### FIXME This part requires many updates for the final design of the report

	plots = []
	imgs = []
	pNo = 1

	### Plot bmask volume
	if bmask is not None:
		#tmpOut = pdf_path.removesuffix(".pdf") + '_plot_' + str(pNo) + '.html'
		#plotWithRef(dfRef, dfSub, 'MUSE_BrainMask', tmpOut)
		#plots.append(_os.path.basename(tmpOut))
		#pNo = pNo + 1

		tmpOut = pdf_path.removesuffix(".pdf") + '_plot_' + str(pNo) + '.png'
		plotWithRef(dfRef, dfSub, 'MUSE_BrainMask', tmpOut)
		plots.append(_os.path.basename(tmpOut))
		imgs.append(tmpOut)
		pNo = pNo + 1

	tables = []
	tNo = 1
	
	### Create data table
	tmpTable = dfSub.describe()
	tmpTable = tmpTable.to_html().replace('<table border="1" class="dataframe">','<table class="table table-striped">') # use bootstrap styling
	idhtml(tmpTable)
	tables.append(tmpTable)

	tmpPatTable = dfPat
	tmpPatTable = tmpPatTable.to_html(index=False).replace('<table border="1" class="dataframe">','<table class="table table-striped">')
	idhtml(tmpPatTable)

	html_out = pdf_path.removesuffix("pdf") + "html"
	writeHtml(plots, tables, tmpPatTable, html_out)
	print('\nTemp html file created: ' + html_out)
	#Convert html to pdf file
	wp(html_out).write_pdf(pdf_path)
	print('\nPDF plot file created: ' + pdf_path)
	#Remove html
	_os.remove(html_out)
	#Remove all plots
	for i in imgs:
		_os.remove(i)

#IF
if __name__ == '__main__':
	_main(nrrd,json,pdf_path)
#ENDIF
