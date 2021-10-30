#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 8 13:49:56 2019

@author: Alberto Fernandez - University of Granada - (alberto@decsai.ugr.es)
"""

import numpy as np

class NominalVariable:
    """
        Representation of a nominal variable.
        Simply a list of all possible values for a given varible
    """
    
    def __init__(self, values):
        """
        A whole "column" is provided in "values", then the unique values are considered
        """
        self.Values_ = np.unique(values)
    
    def getLabelIndex(self, inputValue):
        """
        For a given inputValue, it searches the index in which it appears
        """
        return self.Values_.index(inputValue)