# feature script that models
# antecedent detection through 
# the features collected by Dan and
# Pranav

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
from lib.functions import addPadding



# dummy function to return first
# list entry
def getFirst(x):
    return x[0]



# possible features that can be
# selected
possibleFeatures = {
    "distanceFromSluice": -.92,
    "sluiceCandidateOverlap": 1.37, 
    "backwards": -1.12,
    "WH_gov_npmi": [getFirst,2.65], 
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
#   "sluiceType" not a lambda but a reference to features
}



# normalise feature values 
# through centering
def normalizeVals(features):
    # initialise vars
    first = True
    sluiceCandidateOverlap_min = sluiceCandidateOverlap_max = 0.0

    # loop through candidates and
    # calculate bounds for each
    # feature
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
    
    # loop through candidates and
    # normalise the values given
    # the bounds
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



# modify the features to add logic
# for later use
def modifyFeatures(features):
    # loop over candidates
    for sluiceId, candidateSet in features.items():
        for cand in candidateSet:

            # set backwards feature
            if cand["distanceFromSluice"] >= 0:
                cand["backwards"] = 1
            else:
                cand["backwards"] = 0
                
            # initialise position dependent
            # features and set them
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



# handle correlates in the feature data
# by counting the amount of the different
# lexical items
def handleCorrFeat(sluiceType,key,corrVals, cand):
    mapping = {"Locative": ["location"], "Entity": ["wk"], "Temporal": ["time"], "Which": ["disj"]}

    # check that the sluiceType matches
    # the key
    keyLess = key.replace("Corr", "")
    if sluiceType != keyLess:
        return 0
    
    # calculate the total length
    tot = 0        
    try:
        for lookup in mapping[sluiceType]:
            v = len(corrVals[lookup])
            tot += v
    except KeyError:
        return tot
    
    return tot



# this function is not in use
# since we get the number of features
# as an input into the extracFeatures()
# function
def coefNumber():
    global possibleFeatures

    return len(possibleFeatures) + 1



# extract all features from the data
# given the feature list parameter
def extractFeatures(examples, prepend=False, features=""):
    global possibleFeatures

    # normalize and modify
    # features
    normalizeVals(examples)
    modifyFeatures(examples)

    # create data wrappers
    dataX = []
    dataY = []

    # set appropriate mapping for
    # sluice type, and initialise
    # feature array
    mappings = {"PP": 1, "Degree": 2, "Temporal": 3, "Classificatory": 4, "Focus": 5, "Possessor": 6, "Entity": 7, "Passive": 8, "Reason": 9, "Locative": 10, "Which": 11, "Manner": 12, "None": 13 }
    features = [] if features == "" else features.split(",")

    # loop through all candidates
    maxLength = 0
    for key, candidateSet in examples.items(): 

        # holds features for the
        # current candidate set
        setData = []
        length = 0
        for cand in candidateSet:
            # get candidate information
            sluiceType = cand["sluiceType"]
            corrVals = cand["corrEls"]

            # set all features if need be
            if "sluiceType" in features:
                setData.append(mappings[str(sluiceType)])
            for lkey, factor in possibleFeatures.items():
                if lkey in features:
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

            # add antecedents position to the
            # target labels for multiclassification
            if cand["isAntecedent"]:
                dataY.append(length)

            length += 1

        # add data and update length
        dataX.append(setData)
        if length > maxLength:
            maxLength = length

    # pad data and return
    dataX = addPadding(dataX, maxLength * len(features), prepend)
    return dataX, dataY
