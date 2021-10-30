#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:59:30 2019
Modified on Wed May 29 17:30:05 2019

@author: Alberto Fernandez - University of Granada - (alberto@decsai.ugr.es)
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:55:23 2019

@author: albertoSaurusRex
"""

import inspect
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import normalize

from DataBase import DataBase
from KnowledgeBase import KnowledgeBase

class ChiRWClassifier(BaseEstimator, ClassifierMixin):  
    """ A classification model which implements the Chi-RW algorithm.

    Parameters
    ----------
    labels : int, default=3
        The number of fuzzy partitions per attribute.
    tnorm : str, default="product"
        The type of t-norm for the aggregation of the membership and association 
        degrees. Possible values = {"product","minimum"}
    rw: str, default="pcf"
        The type of rule weight (confidence degree). Possible values = 
        {"cf","pcf","none"}
    frm : str, default="wr"
        The fuzzy reasoning method or inference, between winning rule and 
        additive combination. Possible values = {"wr","ac"}

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self, labels=3, tnorm="product", rw="pcf", frm="wr", RW_tsh=0):
        """
        Called when initializing the classifier
        only product tnorm is implemented throughout the source code
        so tnorm param is useless
        """
        print("Initializing classifier:\n")
        """
        self.labels = labels
        self.tnorm = tnorm
        self.rw = rw
        self.frm = frm
        """
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
            # print("{} = {}".format(arg,val)
        
    def fit(self, X, y,X_Mask):
        """
        This should fit the classifier. All the "work" should be done here.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        
        #The learning part goes here
        dataBase = DataBase(X,self.labels)
        self.kb = KnowledgeBase(X,y,X_Mask,dataBase,self.RW_tsh)
        self.kb.generation()

        # Return the classifier
        return self

    def _meaning(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        return( True if self.kb != None else False )

    def predict(self, X, y=None):
        """ 
        The FRM for the FRBCS.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample according to the FRM of the FRBCS built
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        output = self.kb.predict(X,self.frm) #np array w. class labels
        return output
        
    def predict_proba(self,X,y=None):
        """
        It obtains the normalized fuzzy predictions, i.e. the association degrees to each class

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The normalized fuzzy association for each sample according to the FRM of the FRBCS built
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        output = self.kb.predict_proba(X,self.frm) #np array w. class degrees
        return normalize(output,axis=1,norm="l1")
    