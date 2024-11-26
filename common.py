import pandas as pd
import numpy as np
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error 	
from sklearn.metrics import mean_absolute_percentage_error 	
from tensorflow.keras.models import Sequential
import random
import model as mymodel
import myio
import math
NUMBER_BYTES_PER_KILOBYTE=1024
						
def computeR2(actual,preds):


	#can't have nan values here in computation
	nonNullIxs =  np.isfinite(preds)		
	
	numNonNulls=0
	for f in nonNullIxs:
		#non null element?
		if f:
			numNonNulls = numNonNulls +1
			
	#no null readings?
	if numNonNulls== len(preds):
		r2=r2_score(actual,preds)		
	#there is at least two non-null prediction (R2 for 1 is not appropriate or not prediction not defined)?
	elif numNonNulls >1:
		r2=r2_score(actual[nonNullIxs],preds[nonNullIxs])#subset of  predictions with only non-null values			
	else:
		r2=0			
	
	if r2 <0:
		r2=0 #a model worse than the naive model who predicts mean can be replaced by zero R with 0 R2				
			
	return r2
	
	
def computeMSE(actual,preds):

	#can't have nan values here in computation
	nonNullIxs =  np.isfinite(preds)		
	
	numNonNulls=0
	for f in nonNullIxs:
		#non null element?
		if f:
			numNonNulls = numNonNulls +1
			
	
	#no null readings?
	if numNonNulls== len(preds):
		return mean_squared_error(actual,preds)
	#there is at least one non-null prediction?
	elif numNonNulls >0:		
		return mean_squared_error(actual[nonNullIxs],preds[nonNullIxs])
	else:
		return math.inf #infinite error		


def computeMAPE(actual,preds,smallAbsValLim=0):

	#can't have nan values here in computation
	nonNullIxs =  np.isfinite(preds)		
	
	numNonNulls=0
	for f in nonNullIxs:
		#non null element?
		if f:
			numNonNulls = numNonNulls +1
			
	
	#no null readings?
	if numNonNulls== len(preds):
		return computeMAPEHelper(actual,preds,smallAbsValLim)
	#there is at least one non-null prediction?
	elif numNonNulls >0:		
		return computeMAPEHelper(actual[nonNullIxs],preds[nonNullIxs],smallAbsValLim)
	else:
		return math.inf #infinite error		
		
#compuate mean absolute percentage error (as a ratio, so will need multiply by 100 to convert ratio to percentage)
#smallAbsValLim: the value used to limit the range of small values near 0. By default no smal values are replaced,
#		but when set to a positive number, any prediction or expect value within 0 and that limit, in terms of absoluate
#		value are convert to the respective postiive or negative limit, which ever the value is closest too.
#		Example: with smallValueReplaceLim = 2, any value in the range [0,2) and (-2,0) will be converted to 2 and -2, respectively.
#		This limits large unmeaninful errors due to small actual values
def computeMAPEHelper(actual,pred,smallAbsValLim=0):

	#small value  replacement
	if smallAbsValLim>0:
		if len(actual) != len(pred):
			raise Exception("Cannot compute MAPE, the predictions (length: "+len(actual)+") and expected (length: "+len(pred)+") value should be the same length, but we different lengths.")
			
		n=len(actual)
		sum = 0
		
		lowerLim=-1*smallAbsValLim
		for i in range(n):			
			x = actual[i]
			y = pred[i]
			
			#within positive limit of zero?
			if x >= 0 and x < smallAbsValLim: 
				#convert to smallest positive value
				x=smallAbsValLim
			elif x > lowerLim and x < 0: 
				#convert to largest negative value
				x=lowerLim
				
			#within positive limit of zero?
			if y >= 0 and y < smallAbsValLim: 
				#convert to smallest positive value
				y=smallAbsValLim
			elif y > lowerLim and y < 0: 
				#convert to largest negative value
				y=lowerLim
			
			sum=sum+abs((x-y)/x)
		
		return sum/n
	else:
		return mean_absolute_percentage_error(actual,pred) #no value replacement
#append content of list 2 to list 1
def listAppend(list1,list2):
	for val in list2:
		list1.append(val)
		
#returns list of row indices of dataset randonmly shuffled
def randomRowIndexShuffle(df):
	if isinstance(df,pd.DataFrame):
	
		#number of samples in dataset
		nSamples = len(df.index) 
	elif isinstance(df,int):
		nSamples = df
	else:
		raise Exception ("Cannot perform random shuffle. Invalid type for df Expected pandas dataframe or number of samples")
	
	
	
	

	#randomly shuflle indices	
	indices = []
	for i in range(nSamples):
		indices.append(i)
	random.shuffle(indices)

	return 	indices		
