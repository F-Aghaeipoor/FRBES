#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:35:20 2019
Modified on Mon May 27 18:33:21 2019

@author: Alberto Fernandez - University of Granada - (alberto@decsai.ugr.es)
"""

from FuzzySet import FuzzySet

class FuzzyVariable():
    """ Fuzzy Variable representation. 
    containing <l> linguistic labels (fuzzy sets), being l the number of 
    linguistic labels specified by the user.

    Parameters
    ----------
    labels: int
        The number of fuzzy partitions
    lowerLimit: float 
        The minimum value of the universe of discourse
    upperLimit: float
        The minimum value of the universe of discourse

    Attributes
    ----------
    FuzzySets_ : list, shape (n_labels)
        An Array containing the membership functions.
    MergePoints_ : list, shape (n_labels)
        An Array containing the merge points of the fuzzy sets.        
    """
    
    def __init__(self, labels, lowerLimit, upperLimit):
        self.FuzzySets_ = list()
        self.MergePoints_ = list()
        for label in range(labels):
            #Compute the half of the triangle's base
            halfBase = (upperLimit - lowerLimit) / (labels - 1)
            #We add the half of the triangle's base n times to the lower limit,
            #depending on the linguistic label and the point we want to obtain               
            #(left, mid, right)
            leftPoint = lowerLimit + halfBase * (label - 1)
            midPoint = lowerLimit + halfBase * (label)
            rightPoint = lowerLimit + halfBase * (label + 1)
            if (label == 0):
                leftPoint = midPoint
            elif (label == labels - 1):    
                rightPoint = midPoint

			#We create the fuzzy set
            self.FuzzySets_.append(FuzzySet(leftPoint, midPoint, rightPoint, label))

			#We add the merge point
            if (label > 0):
                self.MergePoints_.append(midPoint - ((midPoint - self.FuzzySets_[label-1].getMidPoint()) / 2))
    
    def get(self,l):
        return self.FuzzySets_[l]
    
    def length(self):
        return len(self.FuzzySets_)
    
    def getLabelIndex(self,inputValue):
        return self.getMaxMembershipFuzzySet(inputValue)
    
    def getMaxMembershipFuzzySet(self,inputValue):
        index = -1

        """
		Since this function is used only in the learning stage,
		we do not compute membership degrees. Instead, we check
		the location of the input value with respect to the point
		where two fuzzy sets merge.
        """
        for i in range(len(self.MergePoints_)):
            if inputValue < self.MergePoints_[i]:
                index = i
                break
            
        #Check last point 
        if (index < 0):
            index = len(self.MergePoints_)

        return index
                

    
    
    