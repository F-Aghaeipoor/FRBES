#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:38:34 2019
Last modified on Thu May 31 12:05:00 2019

@author: Alberto Fernandez - University of Granada - (alberto@decsai.ugr.es)
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


#dataset_name='MB-GE-ER'
#dataset_name='data_myron' 
dataset_name='breast_cancer'
#dataset_name='xor'
# dataset_name='MAGIC'
#dataset_name='LetterRecognitionComplete'
#dataset_name='iris'


def runFRE(X_tr,y_tr,X_tst,y_tst,FI_X,y_nn,prunning_ths):
  topF=3
  X_Mask = getMask(dataset_name,X_tr,FI_X,topF)
  start_time = time.time()
  chi = ChiRWClassifier(labels=3,frm="wr", RW_tsh=prunning_ths)

  chi.fit(X_tr,y_tr,X_Mask)

  y_pred = chi.predict(X_tr)
  acc1=accuracy_score(y_tr,y_pred)*100
  print("The accuracy of Chi-FRBCS model (train) is: ",acc1 )

  y_pred = chi.predict(X_tst)
  acc2=accuracy_score(y_tst,y_pred)*100
  print("\nThe accuracy of Chi-FRBCS model (test) is: ",acc2 )

  #Only for two-class problems
  auc_tst=0
  if len(np.unique(y_tr))==2:
      probas_ = chi.predict_proba(X_tst)
      fpr, tpr, thresholds = roc_curve(y_tst, probas_[:, 1])
      auc_tst = auc(fpr, tpr)
      print("The AUC of Chi-FRBCS model  (test) es: ", auc_tst*100)

  from imblearn.metrics import geometric_mean_score
  GM = geometric_mean_score(y_tst, y_pred)
  print(f'GM of the network on the test data: {GM*100} %')

  t_exec = time.time() - start_time
  hours = int(t_exec / 3600);
  rest = t_exec % 3600;
  minutes = int(rest / 60);
  seconds = rest % 60;

  Fidelity=compute_fidelity(y_nn,y_pred)*100
  print('The Fidelity is the models: ',Fidelity)
  print('--------------------------------')
  print("\nExecution Time: ", hours , ":" , minutes , ":" , '{0:.4g}'.format(seconds))

  cm=confusion_matrix(y_tst, y_pred, normalize='all')
  cmd = ConfusionMatrixDisplay(cm, display_labels=['0','1'])
  cmd.plot()
  NR=chi.kb.NR
  TRL=chi.kb.ARL
  NF=topF
  ths=prunning_ths
  return acc1,acc2,auc_tst,GM,NF,NR,TRL,Fidelity,ths
  ##Standard cross-validation is also available
  #scores = cross_val_score(chi, iris.data, iris.target, cv=5,scoring='accuracy')
  #print(scores)




X_tr,y_tr,X_tst,y_tst,FI_X,y_nn =load_data(dataset_name)
runFRE(X_tr,y_tr,X_tst,y_tst,FI_X,y_nn,0)