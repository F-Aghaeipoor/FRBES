#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:32:21 2019
Modified on November 2021


@author: Alberto Fernandez - University of Granada - (alberto@decsai.ugr.es)
@author: Fatemeh Aghaeipoor - Institute for Research in Fundamental Sciences (IPM) - (f.aghaei@ipm.ir)

"""
import itertools
class FuzzyRule:
    """
        Simple fuzzy rule structure
        * antecedents: list of fuzzy/nominal variables
        * classLabel: consequent class
        * ruleWeight: consequent confidence degree
    """
    
    def __init__(self,antecedents,classLabel,ruleWeight,Supp,index,Conf=0):
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
        self.index= index  # index of rule in matchingdegree matrix (r*c)
        self.Conf = Conf
        self.nCovered = 0  # number of examples that are covered by this rule
        self.nWell_Classified = 0  # number of examples that are well classified by this rule (i.e., this rule wins for those)
        self.p = 0   # p percent well performing measure = nWell_Classified/nCovered
        self.toRemove = False


    def getAntecedents(self):
        return self.antecedents

    def getReal_antecedents(self):
        return self.real_antecedents

    def getAntecedent(self, pos):
        return self.antecedents[pos]
    
    def getRW(self):
        # return  self.Conf
        return self.ruleWeight
    
    def getClassLabel(self):
        return self.classLabel
        
    def getLenght(self):
        return len([ i for i in self.antecedents if i!=-1])

    def getRuleByIndex(self,id):
        return self

    def getSubRulesAvailableInRB(self, RB):
        sub = list()
        base = self.getReal_antecedents()
        # print(base)
        for k in range(self.getLenght()-1, 0, -1):
            # print('k = ', k)
            RB_k_lenght = [r for r in RB if r.getLenght() == k]
            subset=list(itertools.combinations(base, k))
            for ri in RB_k_lenght:
                for s in subset :
                    # print(s)
                    # print(ri.getReal_antecedents())
                    if list(s) == ri.getReal_antecedents():
                        # print('yes')
                        sub.append(ri)
        return sub  # must be refine with set including  all sub rules of this rule available in RB

    def printInfo(self):
        print("Rule found: IF ", self.getAntecedents(), "THEN", self.getClassLabel(), ", RL",self.getLenght(), ", RW:", round(self.ruleWeight,3), ', Supp:', self.Supp,
              ', Conf:', self.Conf, ', nCovered:', self.nCovered, ', nWell_Classified:', self.nWell_Classified, ', P:', self.p)

