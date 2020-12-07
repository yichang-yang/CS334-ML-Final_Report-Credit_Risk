import argparse
import numpy as np
import pandas as pd
from numpy import where
from collections import Counter
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score, plot_roc_curve, confusion_matrix, classification_report,precision_recall_fscore_support, fbeta_score
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import time
import os


def file_to_numpy(filename):
	"""
	Read an input file and convert it to numpy
	"""
	df = pd.read_csv(filename)
	return df.to_numpy()

def calc_mistakes(yHat, yTrue):
	num_mistake = 0
	for idx, pred in enumerate(yTrue):
		if pred != yHat[idx]:
			num_mistake += 1
	return num_mistake

def opt_hyperparam_dt(xFeat, y):
	params = {'max_depth': [int(x) for x in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]], 'max_features': ['auto', 'log2'], 'min_samples_split': [x for x in range(2,10)] , 'min_samples_leaf': [x for x in [1, 3, 5]], 'criterion': ['gini', 'entropy']}
	# params = {'max_features': ['auto', 'log2']}
	
	# 	dt_random best estimator DecisionTreeClassifier(criterion='entropy', max_depth=35, max_features='auto', min_samples_split=2, min_samples_leaf=5)
	# number of wrong prediction dt_random: 629
	# accuracy dt_random:  0.9907723905229957
	
	print(params)
	dtc = DecisionTreeClassifier()
	search = GridSearchCV(estimator = dtc, param_grid = params, cv=10, verbose=1, n_jobs=-1)
	search.fit(xFeat, y)
	print("best params: ", search.best_params_)
	search_mean_score = search.cv_results_['mean_test_score']
	print('search_mean_score', search_mean_score)
	print('search.best_estimator_', search.best_estimator_)

	# plt.plot(max_depth['max_depth'], search_mean_score, label='max_depth')
	# plt.xlabel('value for max_depth')
	# plt.ylabel('cross validation accuracy')
	# plt.legend()
	# plt.show()



	# best_params = search.best_params_
	# best_depth_param = best_params['max_depth']
	# best_mls_param = best_params['min_samples_leaf']
	# # best_crit_param = best_params['criterion']
	# grid_mean_score = search.cv_results_['mean_test_score']
	return search #best_depth_param,best_mls_param #,best_crit_param

def opt_hyperparam_rf(xFeat, y):
	# nest = {'n_estimators': [10, 30, 50]}
	# max_depth = {'max_depth': [10, 20, 30]}
	# min_samples_leaf = {'min_samples_leaf': [1, 3, 5]}
	# max_feature = {'max_features': ['sqrt', 'log2'] }
	# criterion = {'criterion': ['gini', 'entropy']}

	# rfc = RandomForestClassifier(max_depth = 20, min_samples_leaf = 20, criterion='gini', max_features='sqrt')
	# nestgrid = GridSearchCV(estimator = rfc, cv = 10, param_grid = nest)
	# nestgrid.fit(xFeat, y)
	# print("best single nest: ", nestgrid.best_params_)
	# nestgrid_mean_score = nestgrid.cv_results_['mean_test_score']
	# print(nest['n_estimator'])
	# print(nestgrid_mean_score)
	# plt.plot(nest['n_estimators'], nestgrid_mean_score, label='number of trees')
	# plt.xlabel('value for number of trees')
	# plt.ylabel('cross validation accuracy')
	# plt.legend()
	# plt.show()

	# rfc = RandomForestClassifier(min_samples_leaf = 20, n_estimators=30, criterion='gini', max_features = 'sqrt')
	# depthgrid = RandomizedSearchCV(estimator = rfc, cv = 10, param_grid = max_depth,n_iter=n_iter_search)
	# depthgrid.fit(xFeat, y)
	# print("best single max depth: ", depthgrid.best_params_)
	# depthgrid_mean_score = depthgrid.cv_results_['mean_test_score']
	# print(max_depth['max_depth'])
	# print(depthgrid_mean_score)
	# plt.plot(max_depth['max_depth'], depthgrid_mean_score, label='max_depth')
	# plt.xlabel('value for max_depth')
	# plt.ylabel('cross validation accuracy')
	# plt.legend()
	# plt.show()

	# rfc = RandomForestClassifier(max_depth = 20, n_estimators=50, criterion='gini', max_features = 'sqrt')
	# leafgrid = GridSearchCV(estimator = rfc, cv = 10, param_grid = min_samples_leaf)
	# leafgrid.fit(xFeat, y)
	# print("best single min sample leaf: ", leafgrid.best_params_['min_samples_leaf'])
	# leafgrid_mean_score = leafgrid.cv_results_['mean_test_score']
	# print(leafgrid_mean_score)
	# plt.plot(min_samples_leaf['min_samples_leaf'], leafgrid_mean_score, label='min_samples_leaf')
	# plt.xlabel('value for min_samples_leaf')
	# plt.ylabel('cross validation accuracy')
	# plt.legend()
	# plt.show() 

	# rfc = RandomForestClassifier(max_depth = 5, min_samples_leaf = 20, n_estimators=30, criterion='gini')
	# featgrid = GridSearchCV(estimator = rfc, cv = 10, param_grid = max_feature)
	# featgrid.fit(xFeat, y)
	# print("best single max feature: ", featgrid.best_params_['max_features'])
	# featgrid_mean_score = featgrid.cv_results_['mean_test_score']
	# print(featgrid_mean_score)
	# plt.plot(max_feature['max_features'], featgrid_mean_score, label='max_features')
	# plt.xlabel('value for max_features')
	# plt.ylabel('cross validation accuracy')
	# plt.legend()
	# plt.show()

	# rfc = RandomForestClassifier(max_depth = 5, min_samples_leaf = 20, n_estimators=30)
	# critgrid = GridSearchCV(estimator = rfc, cv = 10, param_grid = criterion)
	# critgrid.fit(xFeat, y)
	# print("best single criterion: ", critgrid.best_params_['criterion'])
	# critgrid_mean_score = critgrid.cv_results_['mean_test_score']
	# print(critgrid_mean_score)
	# plt.plot(criterion['criterion'], critgrid_mean_score, label='criterion')
	# plt.xlabel('value for criterion')
	# plt.ylabel('cross validation accuracy')
	# plt.legend()
	# plt.show()

	################################################################################
	#   Best parameters from single grid search
	#        max_Depth: 20
	#   max_features: 'sqrt'      min_samples_leaf: 3
	#   nest: 50
	#   
	#   best single max feature:  sqrt     [sqrt, log2]  =  [0.96415566 0.91983584]
	#   best single criterion:  gini   [gini, entropy]  =  [0.96415566 0.9606599 ]
	################################################################################



	n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
	# Number of features to consider at every split
	max_features = ['auto', 'sqrt']
	# Maximum number of levels in tree
	max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
	max_depth.append(None)
	# Minimum number of samples required to split a node
	min_samples_split = [2, 5, 10]
	# Minimum number of samples required at each leaf node
	min_samples_leaf = [1, 2, 4]
	# Method of selecting samples for training each tree
	bootstrap = [True, False]
	random_grid = {'n_estimators': n_estimators,
			   'max_features': max_features,
			   'max_depth': max_depth,
			   'min_samples_split': min_samples_split,
			   'min_samples_leaf': min_samples_leaf,
			   'bootstrap': bootstrap}

	rfc = RandomForestClassifier()
	rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=3, random_state=334, n_jobs = -1)
	rf_random.fit(xFeat, y)
	print(rf_random.best_params_)
	################################################################################################################################################################
	######      300 trees n_iter =100, cv = 3 time taken = 890.1min
	######      {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 40, 'bootstrap': False}
	######      30 trees  n_iter = 10, cv = 3 time taken = 104.1min
	######      {'n_estimators': 2000, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 70, 'bootstrap': False}
	################################################################################################################################################################
	
	return rf_random


def main():
	parser = argparse.ArgumentParser()
	startime = time.time()
	parser.add_argument("xTrain", help="filename for features of the training data")
	parser.add_argument("yTrain", help="filename for labels associated with training data")
	parser.add_argument("xTest", help="filename for features of the test data")
	parser.add_argument("yTest", help="filename for labels associated with the test data")
	parser.add_argument("--seed", default=334, type=int, help="default seed number")
	args = parser.parse_args()
	print("processing files")

	xTrain = file_to_numpy(args.xTrain)
	yTrain = file_to_numpy(args.yTrain)
	xTest = file_to_numpy(args.xTest)
	yTest = file_to_numpy(args.yTest)
	np.random.seed(args.seed)
	print("file processed")
	print("time taken to transform files:" ,time.time() - startime)
	time1 = time.time()

	########################################################################################################
	######  this section of code is used to create a SMOTE dataset
	sm = SMOTE(random_state=123)
	x_res, y_res = sm.fit_resample(xTrain, yTrain)
	xTrain = x_res
	yTrain = y_res
	print('x_res.shape: ', x_res.shape)
	print('y_res len: ', len(y_res))
	print('y_res', y_res)
	########################################################################################################

	scaler = preprocessing.StandardScaler()
	scaler.fit(xTrain)
	nor_xTrain_smote = scaler.transform(xTrain)
	nor_xTest_smote = scaler.transform(xTest)

	########################################################################################################
	####### this block is used to plot the relationship between 2 features: interest rate vs. dti
	# counter = Counter(yTrain.ravel())
	# print(counter)
	# # scatter plot of examples by class label
	# for label, _ in counter.items():
	# 	row_ix = where(yTrain == label)[0]
	# 	plt.scatter(xTrain[row_ix, 1], xTrain[row_ix, 3], label=str(label), alpha=0.5, s=1)
	# plt.ylabel('dti')
	# plt.xlabel('interest rate')
	# plt.legend()
	# plt.show()
	########################################################################################################



	# dt_random = opt_hyperparam_dt(xTrain, yTrain)
	# print("dt_random best estimator", dt_random.best_estimator_)
	# yhat = dt_random.predict(xTest)
	# print(metrics.f1_score(yTest, yhat, average='macro'))
	# num_mistake = calc_mistakes(yhat, yTest.ravel())
	# print("number of wrong prediction dt_random:", num_mistake)
	# print("accuracy dt_random: ", 1 - num_mistake/len(yTest))

	#########################################################################################################
	# best param of decision tree
	# best params:  {'criterion': 'entropy', 'max_depth': 45, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 9}
	# [0.83048609 0.840974   0.8644943  ... 0.97672419 0.97429725 0.9718514 ]
	# number of wrong prediction: 728
	# accuracy: 1 - 0.010679967725372259 = 0.9893200322746277
	#########################################################################################################


	tree = DecisionTreeClassifier(criterion='entropy', max_depth=45, max_features='auto', min_samples_leaf=1, min_samples_split=9)
	# print("tree:", tree)
	tree.fit(nor_xTrain_smote, yTrain.ravel())
	yhat_tree = tree.predict(nor_xTest_smote)
	print("time taken for decision tree fit & predict:" ,time.time() - time1)
	num_mistake = calc_mistakes(yhat_tree, yTest.ravel())
	yscore_tree = tree.predict_proba(nor_xTest_smote)

	print('tree data:' ,precision_recall_fscore_support(yTest, yhat_tree))
	fpr_tree, tpr_tree, thresholds_tree = metrics.roc_curve(yTest.ravel(), yscore_tree[:,1])

	yscore_tree_train = tree.predict_proba(nor_xTrain_smote)
	fpr_tree_train, tpr_tree_train, thresholds_tree_train = metrics.roc_curve(yTrain.ravel(), yscore_tree_train[:,1])
	tree_auc = metrics.auc(fpr_tree, tpr_tree)

	yhat_tree_train = tree.predict(nor_xTrain_smote)

	print("tree f1_score:" ,metrics.f1_score(yTest.ravel(), yhat_tree, average='macro'))
	print('tree train auc:' ,metrics.auc(fpr_tree_train, tpr_tree_train))
	print('tree test auc:' , tree_auc)
	print('tree confusion matrix (tn, fp, fn, tp):\n' , confusion_matrix(yhat_tree, yTest.ravel()))
	# print("number of wrong prediction tree:", num_mistake)
	print("tree accuracy: ", 1 - num_mistake/len(yTest.ravel()))
	print('tree fbeta Score train:', fbeta_score(yTrain.ravel(), yhat_tree_train, beta=0.5))
	print('tree fbeta Score test:', fbeta_score(yTest.ravel(), yhat_tree, beta=0.5))

	plt.title('Receiver Operating Characteristic of DT & RF')
	plt.plot(fpr_tree, tpr_tree, 'b', label = 'Decision Tree, AUC = %0.2f' % tree_auc)

	time2 = time.time()



	# rf_random = opt_hyperparam_rf(xTrain, yTrain.ravel())
	# yhat = rf_random.predict(xTest)
	# num_mistake = calc_mistakes(yhat, yTest)
	# print("number of wrong prediction:", num_mistake)
	# print("accuracy: ", num_mistake/len(yTest))

	# hyper perameter tuning result
	################################################################################################################################################################
	######      300 trees n_iter = 100, cv = 3 time taken = 890.1min
	######      {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 40, 'bootstrap': False}
	######      30 trees  n_iter = 10, cv = 3 time taken = 104.1min
	######      {'n_estimators': 2000, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 70, 'bootstrap': False}
	################################################################################################################################################################
	
	rfc = RandomForestClassifier(n_estimators = 200, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_depth=40, bootstrap=False, n_jobs=-1)
	rfc.fit(nor_xTrain_smote, yTrain.ravel())
	yhat_rfc = rfc.predict(nor_xTest_smote)
	print("time taken for rfc:", time.time()-time2)
	num_mistake = calc_mistakes(yhat_rfc, yTest.ravel())
	print('rf data:' ,precision_recall_fscore_support(yTest, yhat_rfc))
	
	yscore_rfc = rfc.predict_proba(nor_xTest_smote)
	fpr_rfc, tpr_rfc, thresholds_rfc = metrics.roc_curve(yTest.ravel(), yscore_rfc[:,1])
	rfc_auc = metrics.auc(fpr_rfc, tpr_rfc)

	yscore_rfc_train = rfc.predict_proba(nor_xTrain_smote)
	fpr_rfc_train, tpr_rfc_train, thresholds_rfc_train = metrics.roc_curve(yTrain.ravel(), yscore_rfc_train[:,1])
	
	print("rfc f1_score: ", metrics.f1_score(yTest.ravel(), yhat_rfc))
	print('rfc train auc:' ,metrics.auc(fpr_rfc_train, tpr_rfc_train))
	print('rfc test auc:' , rfc_auc)
	print('rfc confusion matrix (tn, fp, fn, tp):\n' ,confusion_matrix(yhat_rfc, yTest.ravel()))
	# print("number of wrong prediction rfc 200:", num_mistake)
	print("rfc accuracy: ", 1 - num_mistake/len(yTest.ravel()))
	yhat_rfc_train = rfc.predict(nor_xTrain_smote)
	print('rfc fbeta Score train:', fbeta_score(yTrain.ravel(), yhat_rfc_train, beta=0.5))
	print('rfc fbeta Score test:', fbeta_score(yTest.ravel(), yhat_rfc, beta=0.5))


	plt.title('Receiver Operating Characteristic of DT & RF')
	plt.plot(fpr_rfc, tpr_rfc, 'g', label = 'Random Forest, AUC = %0.2f' % rfc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()
	################################################################################################################################################################
	# number of wrong prediction rfc 200: 138
	# accuracy rfc 200:  0.9979755006234872
	# time taken for rf 200: 25.79893684387207
	################################################################################################################################################################
	


	####   {'n_estimators': 2000, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 70, 'bootstrap': False}
	# rfc = RandomForestClassifier(n_estimators = 2000, min_samples_split=2, min_samples_leaf=2, max_features='auto', max_depth=70, bootstrap=False, n_jobs=-1)
	# rfc.fit(xTrain, yTrain.ravel())
	# yhat = rfc.predict(xTest)
	# num_mistake = calc_mistakes(yhat, yTest)
	# print("number of wrong prediction 2000:", num_mistake)
	# print("accuracy 2000: ", 1 - num_mistake/len(yTest))
	# print("time taken for rf 2000:" ,time.time() - time3)



	print("total time taken:" ,time.time() - startime)

	# rf_depth ,rf_mls ,rf_crit = opt_hyperparam_rf(xTrain, yTrain.ravel())
	# print("rf_depth: ", rf_depth)
	# print("rf_mls: ", rf_mls)
	# print("rf_crit: ", rf_crit)
	# forest = RF(xTrain, yTrain,  args.criterion, args.maxDepth, args.minLeafSample, args.seed)
	# ytree = tree.predict(xTest)
	# yforest = forest.predict(xTest)
	# tree_mistake = calc_mistakes(ytree, yTest)
	# forest_mistake = calc_mistakes(yforest, yTest)

	# fpr, tpr, thresholds = roc_curve(Ytest, ytree)
	# fpr1, tpr1, thresholds1 = roc_curve(Ytest, yforest)

	# model.train(xTrain, yTrain)
	# yHat = model.mypredict(xTest)
	# num_mistake = calc_mistakes(yHat, yTest)
	# print(num_mistake)


if __name__ == "__main__":
	main()