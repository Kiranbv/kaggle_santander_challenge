# This program implements the santander challenge from kaggle. The goal of this
# project 


# Importing important modules
import os
import numpy
import scipy
import pandas
import sklearn

# Change working directory
wd_present = os.getcwd()
os.chdir('c:\\santander_kaggle')




# -----------------------------------------------------------------------------
# Getting the data
# Read the training and testing files
training_data_full = pandas.read_csv('train.csv')
testing_data_full = pandas.read_csv('test.csv')

# remove columns with zero variance 
names_tr = training_data_full.columns
names_te = testing_data_full.columns

tr_zero = []
for col in training_data_full.columns:
    if(training_data_full[col].std() == 0 and col!='TARGET'):
        tr_zero.append(col)
        
training_data_full.drop(tr_zero, axis=1, inplace='True')
testing_data_full.drop(tr_zero, axis=1, inplace='True')


# eliminate duplicate columns
tr_dupl = []
c = training_data_full.columns
for i in range(len(c)-1):
    v = training_data_full[c[i]].values
    for j in range(i+1,len(c)):
        if numpy.array_equal(v,training_data_full[c[j]].values):
            tr_dupl.append(c[j])

training_data_full.drop(tr_dupl, axis=1, inplace=True)
testing_data_full.drop(tr_dupl, axis=1, inplace=True)

# features removed
feat_rem = tr_zero+tr_dupl



# Put data in to training and testing sets
size_training = training_data_full.shape
print(size_training)
size_testing = testing_data_full.shape
training_data = numpy.zeros([size_training[0],size_training[1]-1])
training_target = numpy.zeros([size_training[0],1])
testing_data = numpy.zeros([size_testing[0],size_testing[1]])

training_target = training_data_full['TARGET'].values
training_data = training_data_full.drop(['ID','TARGET'],axis=1).values
testing_labels = testing_data_full['ID'].values

testing_data = testing_data_full.drop(['ID'],axis=1).values



# cross validation data for testing

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# preprocessing of data

# Applying pca to the data
training_data1 = training_data
testing_data1 = testing_data

from sklearn.preprocessing import normalize,maxabs_scale
normalize(training_data1,axis=1,norm='l1',copy=True)
normalize(testing_data1,axis=1,norm='l1',copy=True)


#from sklearn.decomposition import PCA
#pca_dimred = PCA(n_components = 12)
#pca_dimred.fit(training_data1)
##
#training_data1 = pca_dimred.transform(training_data1)
#testing_data1 = pca_dimred.transform(testing_data1)
#  

# feature selection 
# univariate feature selection
# use chi2
from sklearn.feature_selection import SelectPercentile,SelectKBest
from sklearn.feature_selection import chi2,f_classif

#training_data1 = numpy.concatenate((training_data[:,0:50],training_data[:,251:306]),axis = 1)
#testing_data1 = numpy.concatenate((testing_data[:,0:50],testing_data[:,251:306]),axis = 1)

feat_selec = SelectKBest(f_classif, k=150)
training_data1 = feat_selec.fit_transform(training_data1,training_target)
#fval =  f_classif(training_data1,training_target)
testing_data1 = feat_selec.transform(testing_data1)
size_training = training_data1.shape
size_testing = testing_data1.shape




# use pearson correlation
from scipy.stats import pearsonr
from scipy.stats import spearmanr



# use rfe
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

#estimator = DecisionTreeClassifier()
#feat_selec = RFE(estimator)
#training_data1 = feat_selec.fit_transform(training_data1,training_target)
#testing_data1 = feat_selec.transform(testing_data1)


# use selectfrommodel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier
from sklearn.svm import LinearSVC

#model = SelectFromModel(clf,prefit=True,threshold='median')
nonzer=[]
nonzer1 = []
# use feat importance
clf = AdaBoostClassifier()
clf.fit(training_data1,training_target)
feat_imp = clf.feature_importances_
for i in range(0,size_training[1]):
    if(feat_imp[i]>0):
        nonzer1.append(i)
#
## using correlation
for i in range(0,size_training[1]):
    a = pearsonr(training_data1[:,i],training_target)[0]
    if( a>0.01):
        nonzer.append(i)

training_data11 = numpy.zeros([size_training[0],len(nonzer1)])
testing_data11 = numpy.zeros([size_testing[0],len(nonzer1)])
for i in range(0,len(nonzer1)):
    training_data11[:,i] = training_data1[:,nonzer1[i]]
    testing_data11[:,i] = testing_data1[:,nonzer1[i]]

#
#
#training_data1 = model.transform(training_data1)
#testing_data1 = model.transform(testing_data1)

# kernel transformation
#from sklearn.kernel_approximation import RBFSampler
#rbf_feature = RBFSampler(gamma=1, random_state=1)
#training_data1 = rbf_feature.fit_transform(training_data1) 
#testing_data1 = rbf_feature.fit_transform(testing_data1) 


## data normalization
#from sklearn import preprocessing
#preprocessing. normalize(training_data1, axis=1, copy=True)
#preprocessing. normalize(testing_data1, axis=1, copy=True)


# training and cv data split
from sklearn.cross_validation import train_test_split
training_datachk, cv_datachk, y_trainchk, y_cvchk = train_test_split(training_data1, training_target, test_size=0.4, random_state=42)


#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# building the learning model

## implement basic logistic regression
from sklearn.linear_model import LogisticRegression
#
#classifier = LogisticRegression(penalty='l1',solver='liblinear',max_iter=100)
#classifier.fit(training_data1,training_target)
#classifier.fit(training_datachk,y_trainchk)
#
## fit the training data for chk 
#classifier.fit(training_datachk,y_trainchk)



# Implement random forest classifier
#from sklearn.ensemble import RandomForestClassifier
#randomforest_classifier =  RandomForestClassifier(n_estimators=100,criterion='gini')
#randomforest_classifier.fit(training_data1,training_target)
#randomforest_classifier.fit(training_datachk,y_trainchk)

## Implement ada boost classifier
#from sklearn.ensemble import AdaBoostClassifier
#adaboost_classifier =  AdaBoostClassifier(n_estimators=1000)
##adaboost_classifier.fit(training_data11,training_target)
##
#adaboost_classifier.fit(training_datachk,y_trainchk)


## implement gradient boosting classifier
#from sklearn.ensemble import GradientBoostingClassifier
#gradboost_classifier =  GradientBoostingClassifier(loss='deviance',n_estimators=400)
#gradboost_classifier.fit(training_data1,training_target)
#gradboost_classifier.fit(training_datachk,y_trainchk)



# implement xgboost 
import xgboost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.grid_search import GridSearchCV

num_round = 5

# parameter insights: gamma: if you make gamma larger, performance gets worse.small 
# values are better. gamma of .03 is good.
#xgboost_classifier =  xgboost.XGBClassifier(gamma=.01,missing=numpy.nan,n_estimators=350,max_depth=6,learning_rate=0.018,subsample=0.65,colsample_bytree=0.7,colsample_bylevel=1,reg_alpha=0.3,reg_lambda=2,min_child_weight=4)
#xgboost_classifier = xgboost.XGBClassifier(missing=numpy.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)
#for x in range(1,11):
#    param = {'n_estimators':350,'min_child_weight':6,'max_depth':8, 'eta':0.04,'n_thread':4,'gamma':x, 'subsample':0.85,'colsample_bytree':0.95,'colsample_bylevel':0.45,'reg_alpha':0.05,'reg_lambda':0,'silent':1, 'objective':'binary:logistic'}
#    xgboost_classifier = xgboost.cv(param, dtrain, num_round, nfold=10,metrics={'auc'}, seed = 0)
#    print(param)    
#    print(xgboost_classifier) 
#xgboost_classifier.fit(training_data1,training_target)
#xgboost_classifier.fit(training_datachk,y_trainchk)

clf = xgboost.XGBClassifier(missing=numpy.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)

X_fit, X_eval, y_fit, y_eval= train_test_split(training_data1, training_target, test_size=0.3)

# fitting
clf.fit(training_data1, training_target, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])


# predicting
out_class1= clf.predict_proba(testing_data1)[:,1]


# Finding testing probability
#out_prob = xgboost_classifier.predict_proba(testing_data1)
#out_prob = xgboost_classifier.predict(dtest)

#out_class1 = out_prob[:,1]

# test with cv data
out_prob1 = clf.predict_proba(cv_datachk)
#dcv = xgboost.DMatrix(cv_datachk)
#out_prob1 = xgboost_classifier.predict(dcv)

out_class2 = out_prob1[:,1]
from sklearn.metrics import roc_auc_score
score_tr = roc_auc_score(y_cvchk,out_class2)

# Creating output dataframe to write in csv
output = {'ID':testing_labels.astype(int),
          'TARGET':out_class1}
outputdf = pandas.DataFrame(output)
#------------------------------------------------------------------------------

# test with training data
#out_prob1 = classifier.predict_proba(training_data1)
#out_class2 = out_prob1[:,1]
#from sklearn.metrics import roc_auc_score
#score_tr = roc_auc_score(training_target,out_class2)





# Write to a csv file
outputdf.to_csv('santander_submission.csv',sep = ',',index=False)

 


