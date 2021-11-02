#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:18:11 2019
Modified on Fri May 23 18:20:32 2019


@author: Alberto Fernandez - University of Granada - (alberto@decsai.ugr.es)
"""

from FuzzyVariable import FuzzyVariable
import NominalVariable

class DataBase():
    """ 
    Fuzzy Data Base representation. Includes Array of FuzzyVariable and Nominal Variable (if any).

    Parameters
    ----------
    X: ndarray, shape (n_samples, n_features)
        The input training data
    labels : int
        The number of fuzzy partitions per attribute.

    Attributes
    ----------
    FuzzyLabels_ : list, shape (n_features, n_labels)
        An Array containing all fuzzy variables.
    """
    
    """
    Compute the ranges for each variable / column
        
    * X is the input training data
    * labels are the number of labels
        
    Considers each variable to be either nominal (crisp) of numerical (triangular fuzzy set)
    """
    def __init__(self, X, labels):
    
        self.FuzzyLabels_ = list()
        self.labels = labels
        for column in X.T:
            if type(column[0]) == str:
                fuzzyVar = NominalVariable(column)
            else:
                minValue = min(column)
                maxValue = max(column)
                fuzzyVar = FuzzyVariable(labels,minValue,maxValue)
            
            self.FuzzyLabels_.append(fuzzyVar)

    """
    Obtains the fuzzy labels with highest membership values for the given example
        
    * example is the input example
        
    The antecedent is built as a string to be used as key in a hast table for the rules
    """
    def getRuleFromExample(self,example,example_mask):
        labels = str()
        j=0
        # print(example)
        # print(example_mask)
        for fuzzyVar,inputValue, mask in zip(self.FuzzyLabels_,example,example_mask):

            if mask==1 :
                j+=1
                labels = labels + str(fuzzyVar.getLabelIndex(inputValue))+','
            else:
                labels = labels + str(-1)+','
        # print('len -==============',j)
        return labels

    """
    Computes the fuzzy membership degree according to the fuzzy variables
        
    * variable is the index of the variable
    * label is the index of the fuzzy label 
    * value is the value to be "fuzzyfied"
        
    In case of nominal variable, the output is {0,1} regarding equality
    In case of fuzzy variable, the output is computed with the fuzzy membership function
    """
    def computeMembershipDegree (self, variable, label, value):
    
        if (isinstance(self.FuzzyLabels_[variable],NominalVariable)):
            if self.FuzzyLabels_[variable] == value:
                return 1
            else:
                return 0
        else:   
            return self.FuzzyLabels_[variable][label].getMembershipDegree(value)
    
    """
    Computes the matching degree of the example to the rule. 
        
    * rule is the fuzzy rule (antecedents with Fuzzy/Nominal variables)
    * example is the tuple with the values to be "fuzzyfied"
        
    Product t-norm is used for the whole antecedent. No other t-norm is currently implemented
    """
    def computeMatchingDegree(self,rule,example):
    
        matching = 1.0 

        for i in range(len(example)):
            if (matching == 0):
                break
            else:
                #Product t-norm (only available at present)
                if (isinstance(self.FuzzyLabels_[i],FuzzyVariable)):
                    l = rule.getAntecedent(i)
                    if l!=-1:    # to prevent masked features
                        matching *= self.FuzzyLabels_[i].get(l).getMembershipDegree(example[i])
                else:
                    if self.FuzzyLabels_[i] != example[i]:
                        matching = 0.0
        return matching;        


    def computeMatchingDegree2(self,rule,example):
        
        matching = 1.0 
#        non_masked_Ante=[ind for ind ,val  in enumerate( rule.antecedents) if val != -1]
        non_masked_Ante=rule.real_antecedents
#        print(non_masked_Ante)
        for i in non_masked_Ante:
            if (matching == 0):
                break
            else:
                #Product t-norm (only available at present)
                if (isinstance(self.FuzzyLabels_[i],FuzzyVariable)):
                        l = rule.getAntecedent(i)                  
                        matching *= self.FuzzyLabels_[i].get(l).getMembershipDegree(example[i])
                else:
                        if self.FuzzyLabels_[i] != example[i]:
                            matching = 0.0
        return matching;        

        
        
        
        
        