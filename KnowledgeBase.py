#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 20:41:02 2019
Modified on November 2021

@author: Fatemeh Aghaeipoor - Institute for Research in Fundamental Sciences (IPM) - (f.aghaei@ipm.ir)
@author: Alberto Fernandez - University of Granada - (alberto@decsai.ugr.es)
"""

import numpy as np
from FuzzyRule import FuzzyRule
from Utils import *
import statistics
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
        * FI_X: the importance value of each single input data
        * dataBase: the whole definition of the fuzzy variables
        * RW_tsh : to optimize the rule set
        * RW_measure: namely PCF and PCF_CS

        * matchingDegrees: a numpy array of the matching degrees for each pair {rule, class} 
            this is needed to improve the RW computation
        * ruleBase: a list with all the rules (FuzzyRule class)
        * classLabels: a numpy array with the different class label indices
    
    """
    
    def __init__(self,X,y,FI_X,dataBase,RW_tsh=0,RW_measure='PCF'):
        self.X = X
        self.FI_X = FI_X
        self.y = y
        self.RW_tsh=RW_tsh
        self.RW_measure=RW_measure
        self.dataBase = dataBase
        self.matchingDegrees = np.zeros([1,1],dtype=float)
        self.matchingDegrees_examples = np.zeros([1, 1], dtype=float)
        self.fianl_ruleBase = list()
        self.classLabels = 0
        self.init_parameters()
        self.NF=0
        self.rules_count = 0
        self.maxLen=0

    def init_parameters(self):
        self.X_Mask = list()
        self.ruleBase = list()
        self.totalRL = 0
        self.ARL = 0
        self.NR = 0

    def includeInitialRules(self, ruleBaseTmp):
        self.classLabels = np.unique(self.y)
        self.rules_count = np.zeros(len(self.classLabels))
        self.well_classified = np.zeros(len(self.fianl_ruleBase))
        self.matchingDegrees = np.resize(self.matchingDegrees,(len(ruleBaseTmp),len(self.classLabels)))
        self.matchingDegrees_examples = np.resize(self.matchingDegrees_examples, (len(self.X),len(ruleBaseTmp)))
        for rule in ruleBaseTmp.keys():  #Create Initial rules
            fuzzyRule = FuzzyRule(rule,0,0,0,0,0) # no class, rule weight,Supp,index,Conf yet
            self.ruleBase.append(fuzzyRule)
        
    def computeMatchingDegreesAll(self):
        """
            It obtains the matching degrees of each rule with all the examples,
            and stores the accumulation value for each class label.
            If RW = PCF_CS, the matching Degrees are penalized based on the majority class
        """
        if self.RW_measure == 'PCF':  # PCF RW
            k = -1
            for example,classLabel in zip(self.X,self.y):
                j=-1
                k += 1
                for rule in self.ruleBase:
                    j+=1
                    nu = self.dataBase.computeMatchingDegree2(rule, example)
                    self.matchingDegrees[j][classLabel] += nu
                    self.matchingDegrees_examples[j][k] = nu

        elif self.RW_measure == 'PCF_CS':
            y_ = {}
            max_count = 0
            print('---------------------------')
            for class_label in np.unique(self.y):
                y_[class_label] = self.y.tolist().count(class_label)
                if max_count < y_[class_label]:
                    max_count = y_[class_label]

                print(f"Number of examples in class {class_label}:{ y_[class_label]}")

            cost = {key: max_count / val for key, val in y_.items()}  #cost coefficient to penalize majority class
            k=-1
            for example,classLabel in zip(self.X,self.y):
                j=-1
                k+= 1
                for rule in self.ruleBase:
                    j+=1
                    nu = self.dataBase.computeMatchingDegree2(rule, example) * cost[classLabel]
                    self.matchingDegrees[j][classLabel] += nu
                    self.matchingDegrees_examples[k][j]=nu

    def computeRuleWeight(self,rule,classLabels,classLabels_Info,i):
        """
            It computes the fuzzy confidence (RW) of the rule by Penalized Certainty Factor (PCF with or without cost)
            also the crisp confidence and support of each rule.
            
            * rule: the fuzzy rule
            * classLabels: the possible class labels of this rule (those with MF > 0)
            * classLabels_Info: information related to the accurance of each class label for this certain rule, e.g., {c1:n1,c2:n2,..}.
            * i: the index of the rule (for pre-computed matchingDegrees array)
        """
        ruleWeight = 0.0
        classIndex = -1
        conf=0
        accum = 0.0
        accum = np.sum(self.matchingDegrees[i])
        supp = sum(list(classLabels_Info.values()))
        for classLabel in classLabels:
            matchClass=self.matchingDegrees[i][classLabel] #matchClass
            matchNotClass = accum-matchClass  #matchNotClass
            currentRW = (matchClass - matchNotClass) / accum # PCF = matchClass - matchNotClass / matchClass + matchNotClass
            currentConf = classLabels_Info[classLabel] / supp
            if (currentRW > ruleWeight):
                ruleWeight = currentRW
                conf = currentConf
                classIndex = classLabel
        return classIndex,ruleWeight,supp,conf

    def generation(self,maxLen):
        """
            The whole Rule Base generation by grid covering
        """
        ruleBaseTmp = dict() #= {ant1: {c1:n1,c2:n2,..}, ant2: {c1:n1,c2:n2,..},...}
        self.maxLen = maxLen
        # 1. Initial rule generation
        # 1.1 Rule Generation: Get all possible pairs of <antecedents,consequents>
        for len in range(maxLen):
            # 1.1.1 Select to input features
            self.X_Mask =  getMask(self.X,self.FI_X,len+1)
            NF=((self.X_Mask==1).sum(axis=0)!=0).sum()
            print('---------------------------')
            print('Number of top Features: ', len+1)
            print('Number of Contributing Features: ', NF)
            if self.NF<NF:
                self.NF=NF

            for example,example_mask,label in zip(self.X,self.X_Mask,self.y):
                antecedents = self.dataBase.getRuleFromExample(example,example_mask)
                if antecedents in ruleBaseTmp:
                    classes = ruleBaseTmp[antecedents].keys()
                    if label in classes:
                        ruleBaseTmp[antecedents][label] += 1
                    else:
                        ruleBaseTmp[antecedents][label] = 1
                else:
                    classes = dict()
                    classes[label]=1
                    ruleBaseTmp[antecedents] = classes

            print (f"Number of Initial rules with length {len+1}: {ruleBaseTmp.values().__len__()}" )
        self.includeInitialRules(ruleBaseTmp)  #  Transform the rule base into arrays

        # 1.2. Compute the matching degree of all the examples with all the rules
        if 'PCF' in self.RW_measure :
            self.computeMatchingDegreesAll()

        # 1.3. Compute the rule weight of each rule and solve the conflicts
        i=-1
        ruleWeight=0
        j=0
        self.ruleBase = list() # remove all rules
        for rule,classLabels_Info in ruleBaseTmp.items():
            classLabels = list(classLabels_Info.keys())
            i+=1
            if 'PCF' in self.RW_measure :  # PCF  or  PCF_CS
                classLabel,ruleWeight,supp,conf = self.computeRuleWeight(rule, classLabels,classLabels_Info, i)

            if ruleWeight > 0:
                new_rule=FuzzyRule(rule,classLabel,ruleWeight,supp,i,conf)
                self.ruleBase.append(new_rule)
                # new_rule.printInfo()
            else:
                j+=1
        print('---------------------------')
        print(f"Number of rules Before prunning : {self.ruleBase.__len__()}")
        #
        ## 2. prun coverd rules
        self.ruleBase = self.prun_covered(self.ruleBase)
        print(f"Number of rules After prunning : {self.ruleBase.__len__()}")

        ## 3. best rule selection
        # self.set_three_measures_of_each_rule(self.ruleBase)
        self.ruleBase = self.select_ths(self.ruleBase)
        # self.ruleBase = self.select_topRW_per_Class(self.ruleBase)
        print(f"Number of rules After selection : {self.ruleBase.__len__()}")

        self.fianl_ruleBase = self.ruleBase
        self.printInfo(self.fianl_ruleBase)
        return self.fianl_ruleBase

    def prun_covered(self, RB):
        for k in range(self.maxLen,1,-1):  #  maxlen or maxlen to ..., 2
            RB_Base = list(filter(lambda r: (r.getLenght()== k ) , RB)) #&& r.toRemove == False
            # print(len(RB_Base))
            for base_rule in RB_Base:
                RB_subset = base_rule.getSubRulesAvailableInRB(RB)
                # print(len(RB_subset))
                for sub_rule in RB_subset:
                    if sub_rule.getRW()  >= base_rule.getRW():
                        base_rule.toRemove = True
        return list(filter(lambda r: r.toRemove==False, RB))

    def select_ths(self,RB):
       updated_RB=list()
       for  rule in RB:
           # if rule.p > self.RW_tsh:
           if rule.ruleWeight > self.RW_tsh:
           # if rule.Conf > self.RW_tsh:
               updated_RB.append(rule)
               # rule.printInfo()

       # updated_RB = list()
       # for classLabel in self.classLabels:
       #     RB1 = list(filter(lambda r: r.getClassLabel() == classLabel, RB))
       #     RWs = [i.ruleWeight for i in RB1]
       #     # RW_tsh = statistics.median(RWs)
       #     RW_tsh = statistics.mean(RWs)
       #     print('RW_tsh = ', RW_tsh)
       #     for  rule in RB1:
       #         if rule.ruleWeight >= RW_tsh:
       #             updated_RB.append(rule)

       # updated_RB = list()
       # # self.set_three_measures_of_each_rule(RB)
       # RWs = [i.ruleWeight for i in RB]
       # # RW_tsh = statistics.median(RWs)
       # RW_tsh = statistics.mean(RWs)
       # print('RW_tsh = ', RW_tsh)
       # for  rule in RB:
       #     if rule.ruleWeight >= RW_tsh:
       #         updated_RB.append(rule)

       return updated_RB

    def select_topRW_per_Class(self,RB):
       updated_RB=list()
       for classLabel in self.classLabels:
           RB_This_class = list(filter(lambda r: r.getClassLabel() == classLabel, RB))
           RB_This_class_sorted = sorted(RB_This_class, key=lambda rule: rule.p)   # measure of prunning could be set here  , reverse=True to descending
           updated_RB = updated_RB+RB_This_class_sorted[:round(len(RB_This_class_sorted)/4)]
       RB=updated_RB
       return RB

    def WR(self,example):
        """
            Winning rule inference
            
            Only the single best rule (that with the highest fuzzy matching) 
            determines the class output
        """
        class_degrees = np.ones(len(self.classLabels))*-1000   # we must not have class -1000 in actual labels
        for fuzzyRule in self.fianl_ruleBase:
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
        for fuzzyRule in self.fianl_ruleBase:
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

    def set_three_measures_of_each_rule(self,ruleBase):
            all_counts = list(np.argmax(self.matchingDegrees_examples, axis=1))
            # print(all_counts)
            for fuzzyRule in ruleBase:
                # 1. set_nCovered
                count1 = 0
                for example, example_mask in zip(self.X, self.X_Mask):
                    Covered_Labels = list()
                    Covered_Labels_ = self.dataBase.getRuleFromExample(example, example_mask)
                    for ind, value in enumerate(Covered_Labels_.split(',')[:-1]):
                        Covered_Labels.append(int(value))

                    # print('Covered_Labels ',Covered_Labels)
                    # print('getAntecedents ',fuzzyRule.getAntecedents())
                    if self.compare(fuzzyRule.getAntecedents(),Covered_Labels)==True:
                        count1 += 1
                # print(count1)
                fuzzyRule.nCovered = count1

                # 2.  nWell_Classified :
                n_WR = all_counts.count(fuzzyRule.index)   #number WR of that rule
                fuzzyRule.nWell_Classified = n_WR

                # 3.  set_p :
                fuzzyRule.p = fuzzyRule.nWell_Classified/fuzzyRule.nCovered

                # fuzzyRule.printInfo()

    def compare(self,getAntecedents,Covered_Labels):
        Ant_index = [i for i, val in enumerate(getAntecedents) if val != -1]
        flag = True
        for k in Ant_index:
            if getAntecedents[k] != Covered_Labels[k]:
                flag = False
                break
        return flag

    def rule_count_per_class(self,RB):
        for classLabel in self.classLabels:
            self.rules_count[classLabel] = len(list(filter(lambda r: r.getClassLabel() == classLabel , RB)))

    def printInfo(self,RB):
        RWs=[i.ruleWeight for i in RB]
        RLs = [i.getLenght() for i in RB]
        print('---------------------------')
        self.NR=len(RB)
        print("Size of Final Rule Base: "+str(self.NR))
        self.totalRL = sum(RLs)
        self.ARL=self.totalRL/len(RB)
        print("Average Rule lenght: "+str(self.ARL))
        # print('Number of Prunned rules:', j)
        self.rule_count_per_class(RB)
        for i in self.classLabels:
            print('# Rules in class ',i,': ', self.rules_count[i])
        print(f'(Min RW:{min(RWs)}  , \tMax RW:{ max(RWs)}')
        print('---------------------------')

    def generation2(self, maxLen):
        """
            The whole Rule Base generation by grid covering
        """
        ruleBaseTmp = dict()  # = {ant1: {c1:n1,c2:n2,..}, ant2: {c1:n1,c2:n2,..},...}
        self.X_Mask = getMask(self.X, self.FI_X, maxLen )
        NF = ((self.X_Mask == 1).sum(axis=0) != 0).sum()
        print('Number of top Features: ', maxLen)
        print('Number of Contributing Features: ', NF)
        if self.NF < NF:
            self.NF = NF

        print("Rule Generation")
        # Get all possible pairs of <antecedents,consequents>

        for example, example_mask, label in zip(self.X, self.X_Mask, self.y):
            antecedents = self.dataBase.getRuleFromExample(example, example_mask)
            if antecedents in ruleBaseTmp:
                # print(label)
                classes = ruleBaseTmp[antecedents].keys()
                if label in classes:
                    ruleBaseTmp[antecedents][label] += 1
                else:
                    ruleBaseTmp[antecedents][label] = 1
            else:
                classes = dict()
                classes[label] = 1
                ruleBaseTmp[antecedents] = classes

        print('Number of Initial rules: ', ruleBaseTmp.values().__len__())
        self.includeInitialRules(ruleBaseTmp)  # Transform the rule base into arrays

        # Compute the matching degree of all the examples with all the rules
        if 'PCF' in self.RW_measure:
            print("Computing Matching Degrees All")
            self.computeMatchingDegreesAll()

        # Compute the rule weight of each rule and solve the conflicts
        print("Computing Rule Weights")
        i = -1
        ruleWeight = 0
        j = 0
        self.ruleBase = list()  # remove all rules
        for rule, classLabels_Info in ruleBaseTmp.items():
            classLabels = list(classLabels_Info.keys())
            i += 1
            if 'PCF' in self.RW_measure:  # PCF  or PCF_CS
                classLabel, ruleWeight, supp, conf = self.computeRuleWeight(rule, classLabels, classLabels_Info, i)

            if ruleWeight > 0:
                new_rule = FuzzyRule(rule, classLabel, ruleWeight, supp, i, conf)
                self.rules_count[classLabel] += 1
                self.ruleBase.append(new_rule)
                self.totalRL += new_rule.getLenght()
                new_rule.printInfo()
                # print('RL = ',new_rule.getLenght())
            else:
                j += 1

        # self.ruleBase = self.select_ths(self.ruleBase)
        # self.ruleBase = self.select_topRW_per_Class(self.ruleBase)

        RWs = [i.ruleWeight for i in self.ruleBase]
        print('--------------------------------')
        self.NR = len(self.ruleBase)
        print("Rule Base: " + str(self.NR))
        self.ARL = self.totalRL / len(self.ruleBase)
        print("Average Rule lenght: " + str(self.ARL))
        print('--------------------------------')
        print('Min RW:', min(RWs), '\nMax RW:', max(RWs))
        print('Number of Prunned rules:', j)
        for i in self.classLabels:
            print('# Rules in class ', i, ': ', self.rules_count[i])
        print('--------------------------------')
        return self.ruleBase

    def generation_variantLenght(self,maxLen):
        # generation of rules with different length i.e., 1,2, ... maxLen
        for i in range(maxLen):
            RB_new= self.generation2(i + 1)
            self.set_three_measures_of_each_rule(RB_new)
            self.fianl_ruleBase = self.fianl_ruleBase + RB_new
            self.init_parameters()

        #***************
        # self.fianl_ruleBase = self.select_ths(self.fianl_ruleBase)
        # self.fianl_ruleBase = self.select_topRW_per_Class(self.fianl_ruleBase)
        # ***************

        RWs =  [i.ruleWeight for i in self.fianl_ruleBase]
        RLs=   [i.getLenght() for i in self.fianl_ruleBase]
        print('**********************************Final Report:')
        print('Number of Contributing Features: ', self.NF)
        self.NR=len(self.fianl_ruleBase)
        print("Final Rule Base size : "+str(self.NR))
        self.totalRL=sum(RLs)
        self.ARL=self.totalRL/len(self.fianl_ruleBase)
        print("Average Rule lenght: "+str(self.ARL))
        print('--------------------------------')
        print('Min RW:', min(RWs), '\nMax RW:', max(RWs))
        self.rule_count_per_class()
        for i in self.classLabels:
            print('# Final Rules in class ',i,': ', self.rules_count[i])
        print('--------------------------------')