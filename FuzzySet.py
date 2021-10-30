#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:52:06 2019
Modified on Mon May 13 18:53:43 2019

@author: Alberto Fernandez - University of Granada - (alberto@decsai.ugr.es)
"""

import inspect

class FuzzySet:
    """
        FuzzySet representation: triangular membership function
        Threee points ared needed (left,mid,right) and the index label
    """
    
    
    def __init__(self, leftPoint, midPoint, rightPoint, label):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
    
    def getRightPoint(self):
        return self.rightPoint
    
    def getMidPoint(self):
        return self.midPoint
    
    def getLeftPoint(self):
        return self.rightPoint
    
    def getMembershipDegree (self,value):
        """
            Obtains the membership degree according to the triangular shape
        """
        if (self.leftPoint <= value and value <= self.midPoint):
            if (self.leftPoint == value and self.midPoint == value):
                return 1.0
            else:
                return (value - self.leftPoint) / (self.midPoint - self.leftPoint);
        elif (self.midPoint <= value and value <= self.rightPoint):
            if (self.midPoint == value and self.rightPoint == value):
                return 1.0
            else:
                return (self.rightPoint - value) / (self.rightPoint - self.midPoint)
        else:
            return 0