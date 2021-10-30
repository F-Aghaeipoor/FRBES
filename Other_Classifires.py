# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:05:20 2021

@author: MASNA.CO

"""
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,roc_curve, auc
from sklearn import datasets 

from ChiRWClassifier import ChiRWClassifier
from Utils import *
import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

from sklearn import tree


#dataset_name='MB-GE-ER'
#dataset_name='data_myron' 
#dataset_name='breast_cancer'
#dataset_name='xor'
dataset_name='MAGIC'
#dataset_name='LetterRecognitionComplete'
#dataset_name='iris'



X_tr,y_tr,X_tst,y_tst,FI_X,y_nn =load_data(dataset_name)

X_Mask = getMask(dataset_name,X_tr,FI_X,3)
start_time = time.time()


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_tr,y_tr)

print("Number of Rules: ",clf.get_n_leaves())

y_pred = clf.predict(X_tr)
print("The accuracy train is: ", accuracy_score(y_tr,y_pred)*100)

y_pred = clf.predict(X_tst)
print("\nThe accuracy of test is: ", accuracy_score(y_tst,y_pred)*100)

#Only for two-class problems
if len(np.unique(y_tr))==2:
    probas_ = clf.predict_proba(X_tst)
    fpr, tpr, thresholds = roc_curve(y_tst, probas_[:, 1])
    auc_tst = auc(fpr, tpr)
    print("The AUC of test is: ", auc_tst*100)

t_exec = time.time() - start_time
hours = int(t_exec / 3600);
rest = t_exec % 3600;
minutes = int(rest / 60);
seconds = rest % 60;

print('The Fidelity is the models: ',compute_fidelity(y_nn,y_pred)*100)
print('--------------------------------')
print("\nExecution Time: ", hours , ":" , minutes , ":" , '{0:.4g}'.format(seconds))

cm=confusion_matrix(y_tst, y_pred, normalize='all')
cmd = ConfusionMatrixDisplay(cm, display_labels=['0','1'])
cmd.plot()


##Standard cross-validation is also available
#scores = cross_val_score(chi, iris.data, iris.target, cv=5,scoring='accuracy')
#print(scores)