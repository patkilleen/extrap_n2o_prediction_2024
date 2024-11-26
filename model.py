import traceback
import numpy as np

DEBUGGING_FLAG=False#set to false when run on linux

try:
	from sklearn.model_selection import ParameterSampler
	from sklearn.ensemble import RandomForestRegressor
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import BatchNormalization
	from tensorflow.keras.layers import LSTM
	from tensorflow.keras.layers import Conv2D
	from tensorflow.keras.layers import MaxPooling2D
	from tensorflow.keras.layers import AveragePooling2D
	from tensorflow.keras.layers import Activation
	from tensorflow.keras.layers import Dropout
	from tensorflow.keras.layers import Dense
	from tensorflow.keras.layers import Flatten
	from tensorflow.keras.layers import Input
	from tensorflow.keras.models import Model
	from tensorflow.keras.layers import concatenate
	from tensorflow.keras.callbacks import EarlyStopping
	from tensorflow.keras.layers import Bidirectional
	from tensorflow.keras.layers import AveragePooling1D
	from tensorflow.keras.layers import Layer
	import tensorflow as tf
	from keras.initializers import RandomNormal
	
	from scipy.stats import loguniform
	import keras_tuner as kt
	from tensorflow.keras.optimizers import Adam	
	from keras.layers import Flatten
	from keras.layers.convolutional import Conv1D
	from keras.layers.convolutional import MaxPooling1D
	from sklearn.svm import SVR
	from sklearn.linear_model import LinearRegression
	#from tensorflow.keras.optimizers.legacy import Adam
	#import tensorflow_decision_forests as tfdf
	#from keras import backend as K
except ImportError as e:
	#only consider it error when not debugging and package import failed
	if not DEBUGGING_FLAG:
		print("Failed to import packages: "+str(e)+"\n"+str(traceback.format_exc()))
		exit()

ALG_ZEROR="ZeroR"
ALG_RANDOM_FOREST="RF"
ALG_DEEP_NEURAL_NETWORK="DNN"
ALG_CONVOLUTIONAL_NEURAL_NETWORK="CNN"
ALG_LONG_SHORT_TERM_MEMORY="LSTM"
ALG_SUPPORT_VECTOR_MACHINE="SVM"
ALG_LINEAR_REGRESSION="LR"
deepLearningModelMap={ALG_ZEROR:False,\
						ALG_RANDOM_FOREST:False,\
						ALG_DEEP_NEURAL_NETWORK:True,\
						ALG_CONVOLUTIONAL_NEURAL_NETWORK:True,\
						ALG_LONG_SHORT_TERM_MEMORY:True,\
						ALG_SUPPORT_VECTOR_MACHINE:False,\
						ALG_LINEAR_REGRESSION:False}
modelsWithTensorsMap={ALG_ZEROR:False,\
						ALG_RANDOM_FOREST:False,\
						ALG_DEEP_NEURAL_NETWORK:False,\
						ALG_CONVOLUTIONAL_NEURAL_NETWORK:True,\
						ALG_LONG_SHORT_TERM_MEMORY:True,\
						ALG_SUPPORT_VECTOR_MACHINE:False,\
						ALG_LINEAR_REGRESSION:False}
						
#algorith,year,hyperparameter name
FLStaticHyperParameterMap={ALG_CONVOLUTIONAL_NEURAL_NETWORK:{},\
							ALG_LONG_SHORT_TERM_MEMORY:{},\
							ALG_DEEP_NEURAL_NETWORK:{},\
							ALG_RANDOM_FOREST:{},\
							ALG_SUPPORT_VECTOR_MACHINE:{},\
							ALG_LINEAR_REGRESSION:{},\
							ALG_ZEROR:{}}							
								
class Model:

	#inputShape: the dimensions of an imput sample. So for basic ML, first and only element states number of features.
	def __init__(self,algName,hyperParams,inputShape,model=None):
		self.algName = algName
		self.hyperParams = hyperParams
		self.fittedFlag=False #not trained yet
		self.inputShape = inputShape
		if not (model is None):
			if isinstance(model,Model):
				raise Exception("Expected a raw model but a model wrapper was provided to mymodel.Mode")
				
		if self.algName == ALG_ZEROR:
			self.predictionBuffer =[] #empty list of predictions to avoid re-creating a list each tiem
			self.model = None
		elif self.algName == ALG_RANDOM_FOREST:
			if not hyperParams["bootstrap"]:
				#sklearn RF can't hanve the 'max_samples' defien when bootstrap is false
				#hpSet["max_samples"]=None	
				if "max_samples" in hyperParams:
					hyperParams.pop("max_samples")
					
			if model is None:
				self.model =RandomForestRegressor(**hyperParams)
			else:
				self.model = model
		elif self.algName == ALG_DEEP_NEURAL_NETWORK:			
			
			
			nLayers =hyperParams["number_of_layers"]
					
			layerSizes = []
			for i in range(nLayers):
				layerSizes.append(hyperParams["FC"+str(i)+"_size"])
			
			if model is None:
				self.model = buildDNN(layerSizes,hyperParams["activation"],hyperParams["learning_rate"],inputShape)
			else:
				self.model = model		
		elif self.algName == ALG_LONG_SHORT_TERM_MEMORY:
				
			#determine if we should add fully connected layer at end or not
			if hyperParams["addFCLayerFlag"]:
				fcSize=hyperParams["FC_size"]
			else:
				fcSize=0
			
			if model is None:			
				self.model = buildLSTM(hyperParams["units"],fcSize,hyperParams["activation"],hyperParams["learning_rate"],inputShape)
			else:
				self.model = model		
		elif self.algName == ALG_CONVOLUTIONAL_NEURAL_NETWORK:
			if model is None:			
				self.model = buildCNN(hyperParams["nFeatureMaps"],hyperParams["filterSize"],hyperParams["FC_size"],hyperParams["activation"],hyperParams["learning_rate"],inputShape)
			else:
				self.model = model
		elif self.algName == ALG_SUPPORT_VECTOR_MACHINE:
			if model is None:			
				self.model = SVR(gamma=hyperParams["gamma"],C=hyperParams["C"],kernel = hyperParams["kernel"])
			else:
				self.model = model
		elif self.algName == ALG_LINEAR_REGRESSION:
			if model is None:			
				self.model =LinearRegression()
			else:
				self.model = model
		else:
			raise Exception("Cannot create model. unknown algorithm "+str(algName))
			
		if isDeepLearningModel(self.algName):				
			self.epochs=hyperParams["epochs"]
			self.patience=hyperParams["early_stop_patience"]
			self.batchSize=hyperParams["batchSize"]
			
	#trains the model
	#X is the feature matrix
	#y is target variable list
	def fit(self,train_X,train_y,test_X,test_y):
		self.fittedFlag=True #training complet
		
		if self.algName == ALG_ZEROR:
			self.meanVal = train_y.mean()
		else:
			if isDeepLearningModel(self.algName):
				#stop_early = getDeepLearningEarlyStopping()
				#self.model.fit(X,y,epoch,callbacks=[stop_early]) #deep learning models use the epoch parameter
				#stop_early = getDeepLearningEarlyStopping()
				stop_early = EarlyStopping(monitor='val_loss', patience=self.patience)
				#see https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
				#for details on early stopping
				#verbose = 0 #silent
				#verbose = 2 #show progress at each epoch	
				self.model.fit(train_X,train_y,validation_data=(test_X,test_y),epochs=self.epochs,batch_size=self.batchSize,verbose = 0) #deep learning models use the epoch parameter
			else:
				self.model.fit(train_X,train_y) #basic ML models don't need test data for model fitting
		pass	
		
	#make predictions using trained model (fit must be called first)
	#X is matrix of features
	def predict(self,X):
		
		#can only predict after training the model
		if not self.fittedFlag:			
			raise Exception("Cannot make "+str(self.algName)+" model predictions before training the model (did you forget to call 'fit'?) ")
		
		
		if self.algName == ALG_ZEROR:
			self.predictionBuffer.clear()
			#iterate over each row
			for i in range(len(X)):				
				self.predictionBuffer.append(self.meanVal) #zeroR always predicts mean of target variable in training data
			return np.array(self.predictionBuffer)
		else:
			preds = self.model.predict(X)					
			
			if isDeepLearningModel(self.algName):
				#neural networks output an array of dimenison one for each sample, so 
				#reshape to 1D array (n samples) instead of 2D array  (1 x n samples)
				preds = preds.reshape(len(preds))
			return preds									
				
	def toString(self):
		resStr = "Algorithm: "+self.algName+", fitted: "+str(self.fittedFlag)+". Model: "+str(self.model)+". Hyperparameters: "
		for k in hyperParams:
			resStr = k+": "+str(hyperParams[k])+","				 				
		return resStr
	
#generates 'trials' sets of randomly chosen hyperparameter values using random search, 
def generateHyperparameterSets(algName,temporalResolution,tensorNumberTimeSteps,trials,inputShape,nTrainSamples,nTestSampels):
	
	if trials <=0:
		raise Exception("Expected positive number of random search trials, but received '"+str(trials)+"'")
		
	hyperParamSets=[]
	paramDict={}
	if algName == ALG_ZEROR:		
		for i in range(trials):
			hyperParamSets.append(None) #no hyperparameters for ZeroR		
	elif algName == ALG_RANDOM_FOREST:
		nFeatures=inputShape[0]
		paramDictValueRanges = getRandomForestHypParamSearchSpace(nFeatures)
		hyperParamSets = ParameterSampler(param_distributions =paramDictValueRanges,n_iter = trials)
							
	
	elif algName == ALG_DEEP_NEURAL_NETWORK:#DNN is a MLP with 2-6 hidden layers
		
		paramDict["number_of_layers"]=[]#number layers
		minNumLayers=1
		maxNumLayers=6
		maxLayerSize=256
		for i in  range(minNumLayers,maxNumLayers,1) :#1 to 6 layers
			paramDict["number_of_layers"].append(i)
		for l in range(maxNumLayers):
			layerLName= "FC"+str(l)+"_size"
			paramDict[layerLName]=[]#size of fully connected layer l
			for i in range(8,maxLayerSize,1) : #min layer size 8, steps by 1
				paramDict[layerLName].append(i)
		
		
		hyperParamSets = ParameterSampler(param_distributions =paramDict,n_iter = trials)
	
	elif algName == ALG_LONG_SHORT_TERM_MEMORY:	
		
		paramDict["units"]=[]#number of neurons in LSTM layer
		for i in  range(3,50,1):
			paramDict["units"].append(i)
			
		
		paramDict["addFCLayerFlag"]=[True,False] #whether we add a ending fully connected layer or not
		paramDict["FC_size"]=[]
		for i in  range(8,256,1):
			paramDict["FC_size"].append(i)
	
		hyperParamSets = ParameterSampler(param_distributions =paramDict,n_iter = trials)
	
	elif algName == ALG_CONVOLUTIONAL_NEURAL_NETWORK:
	
	
		paramDict["nFeatureMaps"]=[]
		for i in  range(8,256,1):
			paramDict["nFeatureMaps"].append(i)
		
		paramDict["filterSize"]=[]
		for i in  range(2,tensorNumberTimeSteps,1):
			paramDict["filterSize"].append(i)
		
		
		
		paramDict["FC_size"]=[]
		for i in  range(8,256,1):
			paramDict["FC_size"].append(i)
					
		hyperParamSets = ParameterSampler(param_distributions =paramDict,n_iter = trials)
		
	elif algName == ALG_SUPPORT_VECTOR_MACHINE:
		#https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
		#https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
		paramDict["C"]=loguniform(1e-1, 10**3) #[0.1, 0.5,1,5 10,50,100,500,1000]
		paramDict["gamma"]=loguniform(1e-3, 10**3)
		paramDict["kernel"]=["linear", "poly", "rbf", "sigmoid"]
		hyperParamSets = ParameterSampler(param_distributions =paramDict,n_iter = trials)
	elif algName ==ALG_LINEAR_REGRESSION:
		pass #no hyperparameters for linear regression
	if isDeepLearningModel(algName):
		
		paramDict["early_stop_patience"]=[]#patience
		for i in  range(3,20,1):
			paramDict["early_stop_patience"].append(i)
				
		paramDict["epochs"]=[]
		for i in  range(400,700,50):
			paramDict["epochs"].append(i)				
		
		
		maxBatchSize = int(round(nTrainSamples*0.9)) #batch size is 90% of size of training dataset
		minBatchSize = int(round(nTrainSamples*0.15))#batch size is 15% of size of training dataset
		
		
		paramDict["batchSize"]=[]#patience
		for i in  range(minBatchSize,maxBatchSize,1):
			paramDict["batchSize"].append(i)
				
		paramDict["activation"]=['relu','tanh','sigmoid']
				
		paramDict["learning_rate"]=loguniform(1e-5, 1e-1)
		hyperParamSets = ParameterSampler(param_distributions =paramDict,n_iter = trials)
	return hyperParamSets



def getRandomForestHypParamSearchSpace(nFeatures):

#tfdf.keras.RandomForestModel(
	#hyper params
	#max_depth , default 16, max depth of tree
	#num_trees: the number of decision trees in forest, default 300
	#num_candidate_attributes: how many features used at a node split, default is sqrt(number of input attributes) 
	#bootstrap_size_ratio: default to 1 (100%), so ration of number of samples used to train a tree
	#min_examples: minimum number examples in a node. default 5
	
	paramDict = {}
	paramDict["n_estimators"]=[]#number of trees
	for i in  range(50,500,25) :
		paramDict["n_estimators"].append(i)
	
	paramDict["max_depth"]=[]
	for i in range(8,24,1): #number of trees
		paramDict["max_depth"].append(i)
		
	paramDict["max_depth"].append(None) # no max depth is possible too
	
	paramDict["min_samples_leaf"]=[]
	for i in range(1,15,1): #minimum number of samples to be a leaf
		paramDict["min_samples_leaf"].append(i)
	
	if nFeatures == 1:
		paramDict["max_features"]=[1]
	else:
		paramDict["max_features"]=[]
		for i in range(1,nFeatures,1): #max number of features condiered when splitting
			paramDict["max_features"].append(i)
			
	paramDict["bootstrap"]=[True,False] #whether bootstrap samples is used or not (false means 100% of data set used, when false, then the max_samples ratio of samples used)
	paramDict["max_samples"]=[]
	for i in range(65,95,5): #65, 70,75,...,95%
		paramDict["max_samples"].append(i/100.0) #make sure in form: 0.65, 0.7....
	#paramDict["random_state"]=[42] #we will let the model be different every time, cause it'll allow the multiple executions per trial to have different randomness
	
	
	return paramDict
	
#model: a sequential model that already had the number of layers added to it
def buildDNN(layerSizes,activation,learnRate,inputShape):
	model = Sequential()
	model.add(Input(shape=(inputShape[0])))
		
	nLayers=len(layerSizes)
	
	for i in range(nLayers):
		model.add(Dense(units=layerSizes[i], activation=activation))
	model.add(Dense(1, activation='linear'))	
	model.compile(optimizer=Adam(learning_rate=learnRate),
			#loss='mean_squared_error',metrics=['mae',coeff_determination])
			loss='mean_squared_error',metrics=['mae'])
	return model

def buildCNN(nFeatureMaps,filterSize,fcLayerSize,activation,learnRate,inputShape):

	model = Sequential()
	model.add(Conv1D(nFeatureMaps, filterSize, activation='relu', input_shape=(inputShape[1],inputShape[0])))#(timestep, features)
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(fcLayerSize, activation=activation))
	model.add(Dense(1))#linear activation (default)
	model.compile(optimizer=Adam(learning_rate=learnRate),
			#loss='mean_squared_error',metrics=['mae',coeff_determination])
			loss='mean_squared_error',metrics=['mae'])
	return model


def buildLSTM(units,fcLayerSize,activation,learnRate,inputShape):
	model = Sequential()
	#follow: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
	model.add(LSTM(units,input_shape=(inputShape[1],inputShape[0]))) #(timestep, features)  #THIS IS CRASHING CAUSE NUMPY VERSION INCOMPTABILE WITH TENSOR FLOW... :(
	#add a fully connected layer after LSTM layer?
	
	if fcLayerSize >0:
		model.add(Dense(fcLayerSize, activation=activation))
	model.add(Dense(1, activation='linear'))	
	
	model.compile(optimizer=Adam(learning_rate=learnRate),
			#loss='mean_squared_error',metrics=['mae',coeff_determination])
			loss='mean_squared_error',metrics=['mae'])
	return model




def isDeepLearningModel(algName):
	return deepLearningModelMap[algName]

def isModelWithInputTensors(algName):
	return modelsWithTensorsMap[algName]

	