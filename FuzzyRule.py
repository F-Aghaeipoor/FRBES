#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:32:21 2019
Modified on Mon May 27 18:50:14 2019


@author: Alberto Fernandez - University of Granada - (alberto@decsai.ugr.es)
"""

class FuzzyRule:
    """
        Simple fuzzy rule structure
        * antecedents: list of fuzzy/nominal variables
        * classLabel: consequent class
        * ruleWeight: consequent confidence degree
    """
    
    def __init__(self,antecedents,classLabel,ruleWeight,Supp,Conf=0):
        """
            Creates a rule from already established values
            * antecedents is a list of strings, each of each is simple a number (fuzzy label index)
            * classLabel is the index of the class
            * ruleWeight (between [0,1]) is the computed confidence (fuzzy membership class/fm all)
        """
        self.antecedents = list()
        self.real_antecedents = list()  # to contain antecedents !=-1, the non masked ones.
        for ind,value in enumerate(antecedents.split(',')[:-1]):
            self.antecedents.append(int(value))
            if int(value)!= -1 :
                self.real_antecedents.append(ind)
        self.classLabel = classLabel
        self.ruleWeight = ruleWeight
        self.Supp = Supp
        self.Conf = Conf

    def getAntecedents(self):
        return self.antecedents
    
    def getAntecedent(self, pos):
        return self.antecedents[pos]
    
    def getRW(self):
        return self.ruleWeight
    
    def getClassLabel(self):
        return self.classLabel
        
    def getLenght(self):
        return len([ i for i in self.antecedents if i!=-1])
    