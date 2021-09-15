import os
import sys
import math
import pandas as pd
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
plt.switch_backend('agg')

def checkMakeFolder(fdname):
	if not os.path.isdir(fdname) == True:
		os.system("mkdir "+fdname)

checkMakeFolder("dataSplit")
checkMakeFolder("resultAnalysis")
checkMakeFolder("predictedResult")
checkMakeFolder("hyperParameters")

#set according to your cutoff time
TIME_MAX=200
PENALTY_TIME=200
#use varing PENALTY policy or 1000s fixed
VARING_PENALTY=False
#random seed
np.random.seed(1)
random.seed(1)


#the objective function to minimize, tuning hyperparameters
#relative_score
#max_relative_score
#min_relative_score
#neg_mean_squared_error
def relative_score(y_true, y_pred):
		res=[]
		for i in range(len(y_true)):
			if y_true[i]>y_pred[i]:
				res.append((y_true[i]-y_pred[i])/(y_true[i]))
			else:
				res.append((y_pred[i]-y_true[i])/(y_true[i]))
		return -sum(res)/float(len(res))

def max_relative_score(y_true, y_pred):

		res=[]
		for i in range(len(y_true)):
			if y_true[i]>y_pred[i]:
				res.append((y_true[i]-y_pred[i])/(y_true[i]))
			else:
				res.append((y_pred[i]-y_true[i])/(y_true[i]))
		return -max(res)

score_functions=[make_scorer(relative_score),make_scorer(max_relative_score),"neg_mean_squared_error"]
# here choose "neg_mean_squared_error"
score_f=score_functions[2]


#print solved percentage and avg runtime
def printSvdPercAvgTime(p,runtime):
	#success
	sucs=[]
	for i in runtime:
		if i<TIME_MAX-1:
			sucs.append(i)
	if len(sucs)!=0:
		print(p,float(len(sucs))/len(runtime),"/",float(sum(sucs))/len(sucs))
	else:
		print(p,float(0),"/",float(0))


#get training data, testing data, validation data

#combine all the featurs + runtime_ into allCombine

#"feature_values.csv" contains the features values already selected.
featureFile="feature_values.csv"
featureValue=pd.read_csv('./csv/'+featureFile)
featureValue=featureValue.set_index("instance_id")
allCombine=featureValue.copy()

#encoding/algorithm runtime performance file
algoRTFile="algorithm_runs.csv"
pd1=pd.read_csv('./csv/'+algoRTFile)

algorithmNames=list(set(pd1["algorithm"].values))
algorithmNames=sorted(algorithmNames)

for algo in algorithmNames:
	singleAlg=pd1[pd1["algorithm"]==algo].copy()
	#change index to instance_id
	singleAlg=singleAlg.set_index("instance_id")
	#only save instance_id and running time


	penaltyTime=[]
	#varing penaly policy
	if VARING_PENALTY == True:
		for i in singleAlg.index:
			if float(singleAlg.ix[i].runtime)<(TIME_MAX-1):
				penaltyTime.append(float(singleAlg.ix[i].runtime))
			else:
				#number of timeout
				runtimes=pd1[pd1['instance_id']==i].runtime.values
				toCount=sum([1 if i>TIME_MAX-1 else 0 for i in runtimes])
				penaltyTime.append(TIME_MAX*toCount)

	#fixed 1000 penalty
	else:

		for i in singleAlg["runtime"]:
			if float(i)<(TIME_MAX-1):
				penaltyTime.append(i)
			else:
				penaltyTime.append(float(PENALTY_TIME))


	singleAlg["runtime"]=penaltyTime
	singleAlg=singleAlg[["runtime"]]

	#change "runtime" to "runtime_index" to distinguish different algrithms
	singleAlg.columns=["runtime_"+algo]

	#combine all the runtime of different encodings
	allCombine=allCombine.join(singleAlg)
	#remove duplicated
	allCombine = allCombine[~allCombine.index.duplicated(keep='first')]

allCombine.sort_index()

featureList=allCombine.columns.values[:-len(algorithmNames)]

print("[Feature used]:",featureList)

#drop "na" rows
allCombine=allCombine.dropna(axis=0, how='any')

#drop "?" rows
for feature in featureList[1:]:
	if allCombine[feature].dtypes=="object":
		# delete from the pd1 rows that contain "?"
		allCombine=allCombine[allCombine[feature].astype("str")!="?"]

algs=["runtime_"+algo for algo in algorithmNames]
allRuntime=allCombine[algs]
oracle_value=np.amin(allRuntime.values, axis=1)
oracle_index=np.argmin(allRuntime.values, axis=1)
Oracle_name=[allRuntime.columns[oracle_index[i]].split("_")[1]  for i in range(len(oracle_index))]

allCombine["Oracle_value"]=oracle_value
allCombine["Oracle_name"]=Oracle_name

#shuffle
allCombine=allCombine.iloc[np.random.permutation(len(allCombine))]

# get testing data 20% of the full data:
testIndex=random.sample(range(allCombine.shape[0]), int(allCombine.shape[0]*0.2))

trainIndex=list(range(allCombine.shape[0]))
for i in testIndex:
	if i in trainIndex:
		trainIndex.remove(i)

testSet=allCombine.iloc[testIndex]
trainSetAll=allCombine.iloc[trainIndex]


#split 80% trainset into validSet, trainSet with specified binNum and which bin.
#bin=0, binNum=5.
#the first bin for validing, rest 4bins for training.
def splitTrainValid(datasetX,bin,binNum):
	bin_size=int(math.ceil(len(datasetX)/binNum))
	if bin==0:
		return np.array(datasetX[bin_size:]),np.array(datasetX[:bin_size])
	elif bin==binNum-1:
		return np.array(datasetX[:(binNum-1)*bin_size]),np.array(datasetX[-bin_size:])
	else:
		return np.append(datasetX[:bin_size*(i)],datasetX[bin_size*(i+1):],axis=0),np.array(dataset[bin_size*(i):bin_size*(i+1)])

trainSet,validSet=splitTrainValid(trainSetAll,0,5)

trainSet=pd.DataFrame(trainSet,columns=trainSetAll.columns)
validSet=pd.DataFrame(validSet,columns=trainSetAll.columns)

print("ALL after preprocess:",allCombine.shape)
print("trainAll:",trainSetAll.shape)
print("--trainSet:",trainSet.shape)
print("--validSet:",validSet.shape)
print("testSet:",testSet.shape)


trainSet.to_csv("dataSplit/trainSet.csv",index=False)
validSet.to_csv("dataSplit/validSet.csv",index=False)
testSet.to_csv("dataSplit/testSet.csv",index=False)


#train each model:

#hyperparameters tuning
#grid search
#train and predict, save
bestDepth={}
if os.path.isdir("parameter_pickle"):
    pickleFiles=[pickFile for pickFile in os.listdir('./parameter_pickle') if pickFile.endswith(".pickle")]
    if 'regression_bestDepth.pickle' in pickleFiles:
        with open('./parameter_pickle/regression_bestDepth.pickle', 'rb') as handle:
            bestDepth = pickle.load(handle)

trainResult=trainSet.copy()
validResult=validSet.copy()
testResult=testSet.copy()

for alg in algorithmNames:
	trainSet_X=trainSet.ix[:,featureList].values
	trainSet_y=trainSet["runtime_"+alg].values
	validSet_X=validSet.ix[:,featureList].values
	validSet_y=validSet["runtime_"+alg].values
	testSet_X=testSet.ix[:,featureList].values
	testSet_y=testSet["runtime_"+alg].values

	bestDepthDT=0
	bestDepthRF=0
	bestKNeib=0


	pickleFiles=[pickFile for pickFile in os.listdir('./hyperParameters') if pickFile.endswith(".pickle")]
	if 'regression_bestDepth.pickle' in pickleFiles:
		with open('hyperParameters/regression_bestDepth.pickle', 'rb') as handle:
			bestDepth = pickle.load(handle)
			bestDepthDT,bestDepthRF,bestKNeib=bestDepth.get(alg,(0,0,0))
	if bestKNeib==0 and bestDepthDT==0 and bestDepthRF==0:

		#Load parameter from pickle
		max_depth = range(2, 30, 1)
		dt_scores = []
		for k in max_depth:
			regr_k =tree.DecisionTreeRegressor(max_depth=k)
			loss = -cross_val_score(regr_k, trainSet_X, trainSet_y, cv=10, scoring=score_f)
			dt_scores.append(loss.mean())
		plt.plot(max_depth, dt_scores,label="DT")
		plt.xlabel('Value of depth: Algorithm'+alg)
		plt.ylabel('Cross-Validated MSE')
		bestscoreDT,bestDepthDT=sorted(list(zip(dt_scores,max_depth)))[0]



		max_depth = range(2, 30, 1)
		dt_scores = []
		for k in max_depth:
			regr_k = RandomForestRegressor(max_depth=k)
			loss = -cross_val_score(regr_k, trainSet_X, trainSet_y, cv=10, scoring=score_f)
			dt_scores.append(loss.mean())
		plt.plot(max_depth, dt_scores,label="RF")
		plt.xlabel('Value of depth: Algorithm'+alg)
		plt.ylabel('Cross-Validated MSE')
		bestscoreRF,bestDepthRF=sorted(list(zip(dt_scores,max_depth)))[0]

		max_neigh = range(2, 30, 1)
		kNN_scores = []
		for k in max_neigh:
			kNeigh =KNeighborsRegressor(n_neighbors=k)
			loss = -cross_val_score(kNeigh,trainSet_X, trainSet_y, cv=10, scoring=score_f)
			kNN_scores.append(loss.mean())
		plt.plot(max_neigh, kNN_scores,label="kNN")
		plt.xlabel('Value of depth: regression_'+alg)
		plt.ylabel('Cross-Validated MSE')
		plt.legend()
		bestscoreRF,bestKNeib=sorted(list(zip(kNN_scores,max_neigh)))[0]

		plt.savefig("hyperParameters/regression_"+alg)
		plt.clf()

		bestDepth[alg]=(bestDepthDT,bestDepthRF,bestKNeib)
		with open('hyperParameters/regression_bestDepth.pickle', 'wb') as handle:
			pickle.dump(bestDepth, handle)

	#predict on all trainign validation and test set and save result
	dtModel=tree.DecisionTreeRegressor(max_depth=bestDepthDT)
	dtModel= dtModel.fit(trainSet_X, trainSet_y)
	y_=dtModel.predict(trainSet_X)
	trainResult["DT_"+alg+"_pred"]=y_
	y_=dtModel.predict(validSet_X)
	validResult["DT_"+alg+"_pred"]=y_
	y_=dtModel.predict(testSet_X)
	testResult["DT_"+alg+"_pred"]=y_

	##########
	rfModel=RandomForestRegressor(max_depth=bestDepthRF)
	rfModel= rfModel.fit(trainSet_X, trainSet_y)
	y_=rfModel.predict(trainSet_X)
	trainResult["RF_"+alg+"_pred"]=y_
	y_=rfModel.predict(validSet_X)
	validResult["RF_"+alg+"_pred"]=y_
	y_=rfModel.predict(testSet_X)
	testResult["RF_"+alg+"_pred"]=y_

    #########
	kNeigh =KNeighborsRegressor(n_neighbors=bestKNeib)
	kNeigh= kNeigh.fit(trainSet_X, trainSet_y)
	y_=kNeigh.predict(trainSet_X)
	trainResult["kNN_"+alg+"_pred"]=y_
	y_=kNeigh.predict(validSet_X)
	validResult["kNN_"+alg+"_pred"]=y_
	y_=kNeigh.predict(testSet_X)
	testResult["kNN_"+alg+"_pred"]=y_


trainResult.to_csv("predictedResult/training_result.csv")
validResult.to_csv("predictedResult/validation_result.csv")
testResult.to_csv("predictedResult/testing_result.csv")


#analysis
##solved percent and runtime of
##per algorithm, oracle and ES
##
runtimeIndex=[i for i in trainResult.columns if "runtime" in i]

def drawLine():
	print("------------------------------------------------")

drawLine()
print("trainSet")
print("Indivadual encoding and Oracle performance: ")
#print per algorithm
for alg in runtimeIndex:
	printSvdPercAvgTime(alg.split("_")[1],trainResult[alg])
#print oracle
printSvdPercAvgTime("oracle_portfolio",trainResult.Oracle_value.values)
print("\nEncoding selection performance:")
for mName in "DT,RF,kNN".split(","):

	print(mName)
	encRuntime=[i for i in trainResult.columns if "runtime" in i]
	modelRuntime=[i for i in trainResult.columns if mName in i]
	modelResults=trainResult[encRuntime+modelRuntime].copy()


	#save each instance's predicted runtime of six encoding and the corresponding predicted encoding name
	#(runtime, name)
	#for each instance, sort by runtime, so that we know which is the first predicted one
	modelResultsCopy=modelResults[modelRuntime].copy()
	for i in modelResultsCopy.columns.values:
		modelResultsCopy[i]=[(j,i)for j in modelResultsCopy[i]]
	predictedList=modelResultsCopy.values
	predictedList.sort()

	#the best predicted is the i[0]:(min_runtime, its_name)
	bestpredname=[i[0][1] for i in predictedList]
	bestname=["runtime_"+i.split("_")[1] for i in bestpredname]
	bestruntime=[modelResults.ix[i,bestname[i]]  for i in range(len(modelResults))]
	modelResults["1st_ham"]=bestname
	modelResults["1st_time"]=bestruntime
	printSvdPercAvgTime("1st",bestruntime)

	#the second predicted is the i[1]:(min_runtime, its_name)
	secondpredname=[i[1][1] for i in predictedList]
	secondname=["runtime_"+i.split("_")[1] for i in secondpredname]
	secondruntime=[modelResults.ix[i,secondname[i]]  for i in range(len(modelResults))]
	modelResults["2nd_ham"]=secondname
	modelResults["2nd_time"]=secondruntime
	#printSvdPercAvgTime("2nd",secondruntime)

	#the third predicted is the i[2]:(min_runtime, its_name)
	thirdpredname=[i[2][1] for i in predictedList]
	thirdname=["runtime_"+i.split("_")[1] for i in thirdpredname]
	thirdruntime=[modelResults.ix[i,thirdname[i]]  for i in range(len(modelResults))]
	modelResults["3rd_ham"]=thirdname
	modelResults["3rd_time"]=thirdruntime
	#printSvdPercAvgTime("3rd",thirdruntime)

	modelResults.to_csv(("resultAnalysis/training_result_analysis_"+mName+".csv"))
print("\n")
drawLine()
print("validSet")
print("Indivadual encoding and Oracle performance: ")
for alg in runtimeIndex:
	printSvdPercAvgTime(alg+"",validResult[alg])
printSvdPercAvgTime("oracle_portfolio",validResult.Oracle_value.values)
print("\nEncoding selection performance:")
for mName in "DT,RF,kNN".split(","):
	print(mName)
	encRuntime=[i for i in validResult.columns if "runtime" in i]
	modelRuntime=[i for i in validResult.columns if mName in i]
	modelResults=validResult[encRuntime+modelRuntime].copy()

	modelResultsCopy=modelResults[modelRuntime].copy()
	for i in modelResultsCopy.columns.values:
		modelResultsCopy[i]=[(j,i)for j in modelResultsCopy[i]]
	predictedList=modelResultsCopy.values
	predictedList.sort()

	bestpredname=[i[0][1] for i in predictedList]
	bestname=["runtime_"+i.split("_")[1] for i in bestpredname]
	bestruntime=[modelResults.ix[i,bestname[i]]  for i in range(len(modelResults))]
	modelResults["1st_ham"]=bestname
	modelResults["1st_time"]=bestruntime

	printSvdPercAvgTime("1st",bestruntime)

	secondpredname=[i[1][1] for i in predictedList]
	secondname=["runtime_"+i.split("_")[1] for i in secondpredname]
	secondruntime=[modelResults.ix[i,secondname[i]]  for i in range(len(modelResults))]
	modelResults["2nd_ham"]=secondname
	modelResults["2nd_time"]=secondruntime
	#printSvdPercAvgTime("2nd",secondruntime)

	thirdpredname=[i[2][1] for i in predictedList]
	thirdname=["runtime_"+i.split("_")[1] for i in thirdpredname]
	thirdruntime=[modelResults.ix[i,thirdname[i]]  for i in range(len(modelResults))]
	modelResults["3rd_ham"]=thirdname
	modelResults["3rd_time"]=thirdruntime
	#printSvdPercAvgTime("3rd",thirdruntime)

	modelResults.to_csv(("resultAnalysis/validition_result_analysis_"+mName+".csv"))
print("\n")
print("testSet")
drawLine()
print("Indivadual encoding and Oracle performance: ")
for alg in runtimeIndex:
	printSvdPercAvgTime(alg+"",testResult[alg])
printSvdPercAvgTime("oracle_portfolio",testResult.Oracle_value.values)
print("\nEncoding selection performance: ")
for mName in "DT,RF,kNN".split(","):
	print(mName)
	encRuntime=[i for i in testResult.columns if "runtime" in i]
	modelRuntime=[i for i in testResult.columns if mName in i]
	modelResults=testResult[encRuntime+modelRuntime].copy()

	modelResultsCopy=modelResults[modelRuntime].copy()
	for i in modelResultsCopy.columns.values:
		modelResultsCopy[i]=[(j,i)for j in modelResultsCopy[i]]
	predictedList=modelResultsCopy.values
	predictedList.sort()

	bestpredname=[i[0][1] for i in predictedList]
	bestname=["runtime_"+i.split("_")[1] for i in bestpredname]
	bestruntime=[modelResults.ix[i,bestname[i]]  for i in range(len(modelResults))]
	modelResults["1st_ham"]=bestname
	modelResults["1st_time"]=bestruntime
	printSvdPercAvgTime("1st",bestruntime)

	secondpredname=[i[1][1] for i in predictedList]
	secondname=["runtime_"+i.split("_")[1] for i in secondpredname]
	secondruntime=[modelResults.ix[i,secondname[i]]  for i in range(len(modelResults))]
	modelResults["2nd_ham"]=secondname
	modelResults["2nd_time"]=secondruntime
	#printSvdPercAvgTime("2nd",secondruntime)

	thirdpredname=[i[2][1] for i in predictedList]
	thirdname=["runtime_"+i.split("_")[1] for i in thirdpredname]
	thirdruntime=[modelResults.ix[i,thirdname[i]]  for i in range(len(modelResults))]
	modelResults["3rd_ham"]=thirdname
	modelResults["3rd_time"]=thirdruntime
	#printSvdPercAvgTime("3rd",thirdruntime)

	modelResults.to_csv(("resultAnalysis/testing_result_analysis_"+mName+".csv"))
