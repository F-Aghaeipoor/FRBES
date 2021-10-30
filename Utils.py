# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 08:23:55 2021

@author: Fatemeh Aghaeipoor  -  School of Computer Science, IPM  (f.aghaei@ipm.ir)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets 
from sklearn.feature_selection import SelectKBest,SelectFromModel,RFE,chi2,mutual_info_classif,f_classif



def getMask(dataset_name,X,FI_X, topN=3):
    """
        Find and return the most influencial features for modeling,
        They are determinded based on Features' Importance received from the trained DNN.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input training data
            
        Returns
        -------
        X_Mask : ndarray, shape (n_samples, n_features)
            Mask Values Per Sample, 1 means feature contributs to the model, 0 does not.
            It must be considered here that each sample must have at least one non-mask feature
            
        Attributes
        ----------
        Features' Importance : ndarray(or Tensor), shape (n_samples, n_features)
            An Array containing all importance values variables.
        """    
   
    #X_Mask = np.random.randint(2, size=X.shape)   random mask 0 or 1
#    X_Mask = np.random.choice([0, 1], size=X.shape, p=[.9995, .0005])   random mask with prob of 0 and 1    
    topN_Fe_ind = np.argpartition(-FI_X, topN)   # (N_exampme,topN)
    topN_Fe_ind=np.array(topN_Fe_ind)
    topN_Fe_ind=topN_Fe_ind[:,:topN]   
    X_Mask=np.zeros(FI_X.shape)
    for i, j in enumerate(topN_Fe_ind):
        X_Mask[i,j]=1    
    return X_Mask


def compute_fidelity(y_nn,predicted):   
    fidelity = (y_nn==predicted).sum()/len(y_nn)
    return fidelity
    
def load_data(dataset_name):

    if dataset_name=='data_myron':
        ### Without FS
#        X_tr = pd.read_csv('C:\\Users\\MASNA.CO\\Desktop\\SEARCHES\\Codes\\data\\data_myron\\GSE89843_TEP_Count_Matrix_ruvseq_diff_1_training.csv', index_col=0)
#        y_tr = pd.read_csv('C:\\Users\\MASNA.CO\\Desktop\\SEARCHES\\Codes\\data\\data_myron\\ruvseq_diff_1_training.csv', sep=",", index_col=0)
#        X_tst = pd.read_csv('C:\\Users\\MASNA.CO\\Desktop\\SEARCHES\\Codes\\data\\data_myron\\GSE89843_TEP_Count_Matrix_ruvseq_diff_1_test.csv', index_col=0)
#        y_tst = pd.read_csv('C:\\Users\\MASNA.CO\\Desktop\\SEARCHES\\Codes\\data\\data_myron\\ruvseq_diff_1_test.csv', sep=",", index_col=0)
#        X_tr=np.array(X_tr)
#        y_tr=np.array(y_tr).reshape(-1)
#        X_tst=np.array(X_tst)
#        y_tst=np.array(y_tst).reshape(-1)
        
        ####With FS
        X_tr = pd.read_csv('C:\\Users\\MASNA.CO\\Desktop\\SEARCHES\\Codes\\data\\data_myron\\X_tr.csv', index_col=0)
        y_tr = pd.read_csv('C:\\Users\\MASNA.CO\\Desktop\\SEARCHES\\Codes\\data\\data_myron\\y_tr.csv', sep=",", index_col=0)
        X_tst = pd.read_csv('C:\\Users\\MASNA.CO\\Desktop\\SEARCHES\\Codes\\data\\data_myron\\X_tst.csv', index_col=0)
        y_tst = pd.read_csv('C:\\Users\\MASNA.CO\\Desktop\\SEARCHES\\Codes\\data\\data_myron\\y_tst.csv', sep=",", index_col=0)
        y_tr=np.array(y_tr).reshape(-1)
        y_tst=np.array(y_tst).reshape(-1)

            
        FI_X = pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/data_myron/FI_X.csv',sep=',',)
        y_nn=pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/data_myron/predicted.csv',sep=',',index_col=0)

    elif dataset_name=='MB-GE-ER':
        data = pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/MB-GE-ER/data.csv',sep=',',)
        X = data.drop(['ER_Expr'], axis=1).values
        y = data['ER_Expr'].values
        print(X.shape,y.shape)
        X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.33, random_state=42,shuffle=True)
        FI_X = pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/MB-GE-ER/FI_X.csv',sep=',',)
        y_nn=pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/MB-GE-ER/predicted.csv',sep=',',index_col=0)
   
    elif dataset_name=='breast_cancer' :
        data = pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/breast_cancer_uci/breast_cancer_uci.csv',sep=',',)
        X = data.drop(['diagnosis'], axis=1).values
        y = data['diagnosis'].values
        print(X.shape,y.shape)
        X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.33, random_state=42)
        FI_X = pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/breast_cancer_uci/FI_X.csv',sep=',',)
        y_nn=pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/breast_cancer_uci/predicted.csv',sep=',',index_col=0)
     
    elif dataset_name=='xor':
        data = pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/XOR/data.csv',sep=',',)
        X = data.drop(['xor'], axis=1).values
        y = data['xor'].astype(np.int64).values
        print(X.shape,y.shape)
        X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.33, random_state=42)
        FI_X = pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/XOR/FI_X.csv',sep=',',)
        y_nn=pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/XOR/predicted.csv',sep=',',index_col=0)
        
    elif dataset_name =='MAGIC':
        target_col_name = 'class'
        data = pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/MAGIC/data.csv',sep=',',)
        X = data.drop([target_col_name], axis=1).values
        y = data[target_col_name].values
        print(X.shape,y.shape)
        X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.33, random_state=42)
        FI_X = pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/MAGIC/FI_X.csv',sep=',',)
        y_nn=pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/MAGIC/predicted.csv',sep=',',index_col=0)
    
    elif dataset_name =='LetterRecognitionComplete':
        target_col_name = 'letter'
        data = pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/LetterRecognitionComplete/data.csv',sep=',',)
        X = data.drop([target_col_name], axis=1).values
        y = data[target_col_name].values
        print(X.shape,y.shape)
        X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.33, random_state=42)
        FI_X = pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/LetterRecognitionComplete/FI_X.csv',sep=',',)
        y_nn=pd.read_csv('C:/Users/MASNA.CO/Desktop/SEARCHES/Codes/data/LetterRecognitionComplete/predicted.csv',sep=',',index_col=0)

        
    elif dataset_name=='iris': 
        dataset = datasets.load_iris()
        X,y = dataset.data, dataset.target
        X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.33, random_state=42)
        FI_X = np.ones(X.shape)
        y_nn=np.ones(y_tst.shape)
        
    y_nn= np.array(y_nn).reshape(-1) 
    return X_tr,y_tr,X_tst,y_tst,FI_X,y_nn
