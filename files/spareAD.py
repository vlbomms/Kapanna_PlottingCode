import pandas as pd
import numpy as np
import nrrd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.base import clone
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import plotly.graph_objects as go
from pygam import ExpectileGAM
from pygam.datasets import mcycle
from pygam import LinearGAM
import os as _os

# TODO: Add PMC to reference values
def calculateSpareAD(age,sex,test):
	MUSE_Ref_Values = pd.read_csv('/refs/PENNcombinedharmonized_out.csv')

	MUSE_Ref_Values.drop('SPARE_AD',axis=1,inplace=True)

	MUSE_Ref_Values['Date'] = pd.to_datetime(MUSE_Ref_Values.Date)

	MUSE_Ref_Values = MUSE_Ref_Values.sort_values(by='Date')
	MUSE_Ref_Values_hc = MUSE_Ref_Values.drop_duplicates(subset=['PTID'], keep='first')
	MUSE_Ref_Values_pt = MUSE_Ref_Values[~MUSE_Ref_Values.isin(MUSE_Ref_Values_hc)].dropna()

	MUSE_Ref_Values = MUSE_Ref_Values.drop(['MRID','Study','PTID','SITE','Date'], axis=1, inplace=False)

	# Make dataframe from patient's age, sex, and ROI dictionaries
	test = pd.DataFrame(test, index=[0])
	test.drop('702',axis=1,inplace=True)
	MUSE_Ref_Values.drop('702',axis=1,inplace=True)
	test['Age'] = age
	if sex == "F":
		test['Sex'] = 0
	else:
		test['Sex'] = 1

	# Order test data's columns to those of the reference data
	MUSE_Ref_Values_tmp = MUSE_Ref_Values.drop('Diagnosis_nearest_2.0',axis=1,inplace=False)
	test = test[MUSE_Ref_Values_tmp.columns]

	MUSE_Ref_Values = MUSE_Ref_Values.loc[(MUSE_Ref_Values['Diagnosis_nearest_2.0'] == "CN") | (MUSE_Ref_Values['Diagnosis_nearest_2.0'] == "AD")]
	MUSE_Ref_Values["Diagnosis_nearest_2.0"].replace({"CN": 0, "AD": 1}, inplace=True)

	### Make numpy arrays of zeros to store results for the classifier ###
	distances_MUSE_Ref_Values = np.zeros( 1 )
	predictions_MUSE_Ref_Values = np.zeros( 1 )

	### Get data to be used in classification analysis ###
	# Training data
	X_train_MUSE_Ref_Values = MUSE_Ref_Values.loc[:,MUSE_Ref_Values.columns != "Diagnosis_nearest_2.0"]
	Y_train_MUSE_Ref_Values = MUSE_Ref_Values.loc[:,"Diagnosis_nearest_2.0"].values

	#Testing data
	X_test = test.loc[:,test.columns]

	### Actual train + testing ###
	# scale features
	scaler_MUSE_Ref_Values = preprocessing.MinMaxScaler( feature_range=(0,1) ).fit( X_train_MUSE_Ref_Values )
	X_train_MUSE_Ref_Values_norm_sc = scaler_MUSE_Ref_Values.transform( X_train_MUSE_Ref_Values )
	X_test_MUSE_Ref_Values_norm_sc = scaler_MUSE_Ref_Values.transform( X_test )

	# load classifers
	svc_MUSE_Ref_Values = svm.SVC( probability=True, kernel='linear', C=1 );

	# fit classifers
	svc_MUSE_Ref_Values.fit( X_train_MUSE_Ref_Values_norm_sc, Y_train_MUSE_Ref_Values )

	# get distance and predictions for the test subject
	distances_MUSE_Ref_Values = svc_MUSE_Ref_Values.decision_function( X_test_MUSE_Ref_Values_norm_sc )
	predictions_MUSE_Ref_Values = svc_MUSE_Ref_Values.predict( X_test_MUSE_Ref_Values_norm_sc )

	return distances_MUSE_Ref_Values, predictions_MUSE_Ref_Values

def createSpareADplot(spareAD, dfSub, fig, row, fname):
	ref1 = pd.read_csv('/refs/PENNcombinedharmonized_out.csv')
	ref2 = pd.read_csv('/refs/ABC+PMCdata.csv')
	ref2['Sex'].replace(0, 'F',inplace=True)
	ref2['Sex'].replace(1, 'M',inplace=True)

	allref = pd.concat([ref1,ref2], axis=0, ignore_index=True)

	# Age and sex filter
	allref = allref[allref.Sex == dfSub['Sex'].values[0]]
	lowlim = int(dfSub['Age'].values[0]) - 5
	uplim = int(dfSub['Age'].values[0]) + 5
	allref = allref.dropna(subset=['SPARE_AD'])
	limRef = allref.loc[(allref['Age'] >= lowlim) & (allref['Age'] <= uplim)]	

	dfSub['SPARE_AD'] = spareAD

	### Get AD reference values
	ADRef = limRef.loc[limRef['Diagnosis_nearest_2.0'] == 'AD']

	### Get CN reference values
	CNRef = allref.loc[allref['Diagnosis_nearest_2.0'] == 'CN']

	### Get CN data points to create GAM lines with
	X = CNRef.Age.values.reshape([-1,1])
	y = CNRef['SPARE_AD'].values.reshape([-1,1])

	### Get age-limited CN reference values
	CNRef = limRef.loc[limRef['Diagnosis_nearest_2.0'] == 'CN']

	##############################################################
	## Fit expectiles
	# fit the mean model first by CV
	gam50 = ExpectileGAM(expectile=0.5).gridsearch(X, y)

	# and copy the smoothing to the other models
	lam = gam50.lam

	# fit a few more models
	gam90 = ExpectileGAM(expectile=0.90, lam=lam).fit(X, y)
	gam75 = ExpectileGAM(expectile=0.75, lam=lam).fit(X, y)
	gam25 = ExpectileGAM(expectile=0.25, lam=lam).fit(X, y)
	gam05 = ExpectileGAM(expectile=0.05, lam=lam).fit(X, y)
	
	XX = gam50.generate_X_grid(term=0, n=100)
	XX90 = list(gam90.predict(XX).flatten())
	XX75 = list(gam75.predict(XX).flatten())
	XX50 = list(gam50.predict(XX).flatten())
	XX25 = list(gam25.predict(XX).flatten())
	XX05 = list(gam05.predict(XX).flatten())
	XX = list(XX.flatten())

	fig.append_trace( go.Scatter(mode='markers', x=ADRef.Age.tolist(), y=ADRef['SPARE_AD'].tolist(), marker=dict(color='MediumVioletRed', size=5), opacity=0.2, name='AD Reference', showlegend=False),row, 1)
	fig.append_trace( go.Scatter(mode='markers', x=CNRef.Age.tolist(), y=CNRef['SPARE_AD'].tolist(), marker=dict(color='MediumBlue', size=5), opacity=0.2, name='CN Reference', showlegend=False),row, 1)
	fig.append_trace( go.Scatter(mode='lines', x=XX, y=XX90, marker=dict(color='MediumPurple', size=10), name='90th percentile', showlegend=False),row,1)
	fig.append_trace( go.Scatter(mode='lines', x=XX, y=XX75, marker=dict(color='Orchid', size=10), name='75th percentile', showlegend=False),row,1)
	fig.append_trace( go.Scatter(mode='lines', x=XX, y=XX50, marker=dict(color='MediumVioletRed', size=10), name='50th percentile', showlegend=False),row,1)
	fig.append_trace( go.Scatter(mode='lines', x=XX, y=XX25, marker=dict(color='Orchid', size=10), name='25th percentile', showlegend=False),row,1)
	fig.append_trace( go.Scatter(mode='lines', x=XX, y=XX05, marker=dict(color='MediumPurple', size=10), name='5th percentile', showlegend=False),row,1)
	fig.append_trace( go.Scatter(mode='markers', x=dfSub.Age.tolist(), y=dfSub.SPARE_AD.tolist(),marker=dict(color='Red', symbol = 'circle-cross-open', size=16,line=dict(color='MediumPurple', width=3)), name='Patient', showlegend=False),row,1)

	fig.update_xaxes(range=[lowlim, uplim])	

	fig.add_annotation(xref='x domain',yref='y domain',x=0.01,y=0.9, text='SPARE AD: ' + str(round(spareAD[0],2)), showarrow=False,row=row, col=1)

	fig.update_layout(legend=dict(x=.93,xanchor='right',y=0),margin_b=0,margin_l=0,margin_r=0,margin_t=20)
	fig.write_image(fname, scale=1, width=800, height=600)

	#fig.update_layout(yaxis_title="SPARE AD", legend=dict(orientation="h"),margin_b=0,margin_l=0,margin_r=0,margin_t=25)
	#tmpOut = '/tmpfolder/' + _os.path.basename(pdf_path.removesuffix(".pdf") + '_spareADplot.png')
	#original height = 570
	#fig.write_image(tmpOut, scale=1, width=650, height=210)

	#return tmpOut