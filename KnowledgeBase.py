#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 20:41:02 2019
Modified on Tue May 28 16:47:00 2019

@author: Alberto Fernandez - University of Granada - (alberto@decsai.ugr.es)
"""

import numpy as np
from FuzzyRule import FuzzyRule

class KnowledgeBase:
    """
    The whole Knowledge Base is represented here
    
    Parameters
    ----------
        * X: numpy 
            The training set input values
        * y: numpy 
            The training set output class labels
        * dataBase: DataBase 
            The definition of the Fuzzy Variables
    
    Attributes
    ----------
        * X: the training set input values
        * y: the trainign set output class labels
        * dataBase: the whole definition of the fuzzy variables
        * matchingDegrees: a numpy array of the matching degrees for each pair {rule, class} 
            this is needed to improve the RW computation
        * ruleBase: a list with all the rules (FuzzyRule class)
        * classLabels: a numpy array with the different class label indices
    
    """
    
    def __init__(self,X,y,X_Mask,dataBase,RW_tsh=0):
        self.X = X
        self.X_Mask = X_Mask
        self.y = y
        self.RW_tsh=RW_tsh
        self.dataBase = dataBase
        self.matchingDegrees = np.zeros([1,1],dtype=float)
        #print(self.matchingDegrees)
        self.ruleBase = list()
        self.totalRL=0
        self.classLabels = 0
        self.rules_count =  0
    def includeInitialRules(self, ruleBaseTmp):
        self.classLabels = np.unique(self.y)
        self.rules_count = np.zeros(len(self.classLabels))
        self.matchingDegrees = np.resize(self.matchingDegrees,(len(ruleBaseTmp),len(self.classLabels)))
    
        #Create rules
        for rule in ruleBaseTmp.keys():
            fuzzyRule = FuzzyRule(rule,0,0) #no class yet, no rule weight yet
            self.ruleBase.append(fuzzyRule)
        
    def computeMatchingDegreesAll(self):
        """
            It obtains the matching degrees of each rule with all the examples,
            and stores the accumulation value for each class label
        """
        for example,classLabel in zip(self.X,self.y):
            j=-1
            for rule in self.ruleBase:
                j+=1
                nu = self.dataBase.computeMatchingDegree2(rule, example)
                self.matchingDegrees[j][classLabel] += nu

    def computeRuleWeight(self,rule,classLabels,i):
        """
            It computes the confidence of the rule by Penalized Certainty Factor 
            (no other is supported yet)
            
            * rule: the fuzzy rule
            * classLabels: the possible class labels of the rule (those with MF > 0)
            * i: the index of the rule (for pre-computed matchingDegrees array)
        """
        ruleWeight = 0.0
        classIndex = -1
        accum = 0.0
        accum = np.sum(self.matchingDegrees[i])

        for classLabel in classLabels:
            sumOthers = accum-self.matchingDegrees[i][classLabel];
            currentRW = (self.matchingDegrees[i][classLabel] - sumOthers) / accum #P-CF
            if (currentRW > ruleWeight):
                ruleWeight = currentRW
                classIndex = classLabel
        return classIndex,ruleWeight

        
    def generation(self):
        """
            The whole Rule Base generation by grid covering
        """
        ruleBaseTmp = dict() #initial hash table to avoid repetitions

        print("Rule Generation")
        #Get all possible pairs of <antecedents,consequents>

        for example,example_mask,label in zip(self.X,self.X_Mask,self.y):
#
            antecedents = self.dataBase.getRuleFromExample(example,example_mask)
            if antecedents in ruleBaseTmp:
                classes = ruleBaseTmp[antecedents]
                if label not in classes:
                    ruleBaseTmp[antecedents].append(label)
#                    SP[antecedents]+=1
#                    CP[antecedents] should +=1 for label
            else:
                classes = list()
                classes.append(label)
                ruleBaseTmp[antecedents] = classes
#                SP[antecedents]=1
#                CP[antecedents]= 1 for label and 0 for other class 
                
        ###To Do 
        # Dictionary of Dictionary to Count occurance of (Ant label)

        print("Computing Matching Degrees Rule")
        """
            Transform the rule base into arrays
        """
        self.includeInitialRules(ruleBaseTmp);

        print("Computing Matching Degrees All")
        """
            Compute the matching degree of all the examples with all the rules
        """
        self.computeMatchingDegreesAll();

        print("Computing Rule Weights")
        """
            Compute the rule weight of each rule and solve the conflicts
        """
        i=-1
        self.ruleBase = list() #remove all rules
        for rule,classLabels in ruleBaseTmp.items():
            i+=1
            classLabel,ruleWeight = self.computeRuleWeight(rule,classLabels,i)
            if ruleWeight > self.RW_tsh:  
                new_rule=FuzzyRule(rule,classLabel,ruleWeight)
                self.rules_count[classLabel]+=1
                self.ruleBase.append(new_rule)
                self.totalRL += new_rule.getLenght()
#                print("Rule found: IF ",rule,"THEN",classLabel,"RW:",ruleWeight)
        RWs=[i.ruleWeight for i in self.ruleBase]
        print('--------------------------------')
        print("Rule Base: "+str(len(self.ruleBase)))
        print("Average Rule lenght: "+str(self.totalRL/len(self.ruleBase)))
        print('--------------------------------')
        print('Min RW:', min(RWs), '\nMax RW:', max(RWs))

        for i in self.classLabels:
            print('# Rules in class ',i,': ', self.rules_count[i])
        print('--------------------------------')

        
    def WR(self,example):
        """
            Winning rule inference
            
            Only the single best rule (that with the highest fuzzy matching) 
            determines the class output
        """
        class_degrees = np.zeros(len(self.classLabels))
        for fuzzyRule in self.ruleBase:
            degree = self.dataBase.computeMatchingDegree2(fuzzyRule,example)
            degree *= fuzzyRule.getRW()
            class_label = fuzzyRule.getClassLabel()
            if class_degrees[class_label] < degree:
                class_degrees[class_label] = degree
        
        return class_degrees
    
    def AC(self,example):
        """
            Additive combination inference
            
            All rules take course in the decision of the class label
        """
        classDegrees = np.zeros(len(self.classLabels))
        for fuzzyRule in self.ruleBase:
            degree = self.dataBase.computeMatchingDegree2(fuzzyRule,example)
            degree *= fuzzyRule.getRW()
            classDegrees[fuzzyRule.getClassLabel()] += degree
                
        return classDegrees
    
    def classification(self,example,frm):
        if frm == "wr":
            return self.WR(example)
        else:
            return self.AC(example)
        
    def predict(self,X,frm):
        prediction = np.zeros(X.shape[0],dtype=int)
        for i in range(X.shape[0]):
            prediction[i] = np.argmax(self.classification(X[i],frm))
        return prediction
    
    def predict_proba(self,X,frm):
        prediction = np.zeros((X.shape[0],len(self.classLabels)),dtype=float)
        for i in range(X.shape[0]):
            prediction[i] = self.classification(X[i],frm) #probabilities (unnormalized)
        return prediction