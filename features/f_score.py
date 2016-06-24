#!/usr/bin/python
import csv
import json
from collections import defaultdict, Counter
import pdb
import sys, traceback
import copy
import random
import math

import sys
# import parent libraries
sys.path.insert(0, '.')

from lib.data import loadData



def getFirst(x):
    return x[0]



Lambdas = {
    "distanceFromSluice": -.92,
    "sluiceCandidateOverlap": 1.37, 
    "backwards": -1.12,
    "WH_gov_npmi": [getFirst,2.65], #[first] 
    "containsSluice": -4.44,
    "isDominatedBySluice": -10,
    "isInRelClause": -1.40,
    "isInParenthetical": -2.4,
    "coordWithSluice": -.46,
    "immedAfterCataphoricSluice": 2.13,
    "afterInitialSluice": -1.85,
    "sluiceInCataphoricPattern": -1.03,
    "LocativeCorr": 0,
    "EntityCorr": 0,
    "TemporalCorr": 0,
    "DegreeCorr": 0,
    "WhichCorr": 0 
}



def normalizeVals(features):
    first = True
    sluiceCandidateOverlap_min = sluiceCandidateOverlap_max = 0.0
    for sluiceId, candidateSet in features.items():
        for c in candidateSet:
            if first:
                WH_gov_npmi_min =  WH_gov_npmi_max = c["WH_gov_npmi"][0]
                first = False
            else:
                if c["WH_gov_npmi"][0] > WH_gov_npmi_max:
                    WH_gov_npmi_max = c["WH_gov_npmi"][0]
                if c["WH_gov_npmi"][0] < WH_gov_npmi_min and c["WH_gov_npmi"][0] > 0: 
                    WH_gov_npmi_min = c["WH_gov_npmi"][0]
                if c["sluiceCandidateOverlap"] > sluiceCandidateOverlap_max:
                    sluiceCandidateOverlap_max =  c["sluiceCandidateOverlap"]
                if c["sluiceCandidateOverlap"] < sluiceCandidateOverlap_min:
                    sluiceCandidateOverlap_min =  c["sluiceCandidateOverlap"]
                
    for sluiceId, candidateSet in features.items():
        for c in candidateSet:
            if c["WH_gov_npmi"][1] == 'MISSING': #nothing left here, set it to the minimum
                c["WH_gov_npmi"][0] = 0
            else:
                c["WH_gov_npmi"][0] = (c["WH_gov_npmi"][0] - WH_gov_npmi_min) / (WH_gov_npmi_max - WH_gov_npmi_min)
            if c["WH_gov_npmi"][0] <0:
                pdb.set_trace()
            if c["sluiceCandidateOverlap"] > 0:
                c["sluiceCandidateOverlap"] = float(c["sluiceCandidateOverlap"] - sluiceCandidateOverlap_min) / float(sluiceCandidateOverlap_max - sluiceCandidateOverlap_min);



def modifyFeatures(features):
    for sluiceId, candidateSet in features.items():
        for cand in candidateSet:
            if cand["distanceFromSluice"] >= 0:
                cand["backwards"] = 1
            else:
                cand["backwards"] = 0
                
            cand["immedAfterCataphoricSluice"] = 0
            cand["afterInitialSluice"] = 0
            cand["sluiceInCataphoricPattern"] = 0
                
            if "do n't know why , but" in cand["sluiceLineText"] or (cand["sluiceInBut"] and cand["sluiceNegated"]):
                cand["sluiceInCataphoricPattern"] = 1
                if cand["distanceFromSluice"] in [1,2] and cand["backwards"]==1:
                    cand["immedAfterCataphoricSluice"] = 1
                        
            if cand["sluiceIsInitial"] and not cand["sluiceInCataphoricPattern"] and cand["sentence"] == 1:
                cand["afterInitialSluice"] = 1
                            
            if cand["distanceFromSluice"] < 0:
                cand["distanceFromSluice"] *= -1



def featureNumber():
    global Lambdas

    return len(Lambdas) + 1



def extractFeatures(examples):
    global Lambdas

    def handleCorrFeat(sluiceType,key,corrVals, cand):
        mapping = {"Locative": ["location"], "Entity": ["wk"], "Temporal": ["time"], "Which": ["disj"]}

        keyLess = key.replace("Corr", "")
        if sluiceType != keyLess:
            return 0
            
        tot = 0        
        try:
            for lookup in mapping[sluiceType]:
                v = len(corrVals[lookup])
                tot += v
        except KeyError:
            return tot
        
        return tot

    normalizeVals(examples)
    modifyFeatures(examples)

    dataX = []
    dataY = []
    mappings = {"PP": 1, "Degree": 2, "Temporal": 3, "Classificatory": 4, "Focus": 5, "Possessor": 6, "Entity": 7, "Passive": 8, "Reason": 9, "Locative": 10, "Which": 11, "Manner": 12, "None": 13 }
    
    maxLength = 0
    for key, candidateSet in examples.items(): 
        setData = []
        length = 0
        for cand in candidateSet:
            sluiceType = cand["sluiceType"]
            corrVals = cand["corrEls"]
            
            setData.append(mappings[str(sluiceType)])

            for lkey, factor in Lambdas.items():
                if "WH_gov_npmi" in lkey:
                    setData.append(cand[lkey][0])
                elif "Corr" not in lkey:
                    if cand[lkey] == 1:
                        setData.append(1)
                    elif cand[lkey] == 0:
                        setData.append(0)
                    else:
                        setData.append(cand[lkey])
                else:
                    c = handleCorrFeat(sluiceType, lkey, corrVals, cand)
                    setData.append(c)

            if cand["isAntecedent"]:
                dataY.append(length)

            length += 1


        dataX.append(setData)

        if length > maxLength:
            maxLength = length


    for example in dataX:
     if len(example) < maxLength * featureNumber():
        for i in range(maxLength * featureNumber() - len(example)):
            example.append(0.0)

    #print maxLength, "-", len(dataX[0]), featureNumber()
    return dataX, dataY






