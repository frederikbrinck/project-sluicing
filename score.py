# this code was originally created by Pranav Anand at UCSC
# and Daniel Hardt at CBS as a part of their sluicing paper;
# the current form is an adapted and heavily modified version
# that allows for further feature analysis.
import csv
import json
from collections import defaultdict, Counter
import pdb
import sys, traceback
import copy
import random
import math

from lib.data import loadData
from lib.functions import getAntecedent

def getFirst(x):
    return x[0]

# best parameter intialisation
Lambdas = {
    # "lmScore": 3.41,
    # "ngram2": 0.58,
    # "ngram3": 2.02,
    # "w2vMaxSimilarity": -0.31,
    # "w2vMainPredicate": 1,
    "w2vPredicates": 2.1,
    # "distanceFromSluice": -.92,
    # "sluiceCandidateOverlap": 1.37, 
    # "backwards": -1.12,
    # "WH_gov_npmi": [getFirst,2.65], #[first] 
    # "containsSluice": -4.44,
    # "isDominatedBySluice": -10,
    # "isInRelClause": -1.40,
    # "isInParenthetical": -2.4,
    # "coordWithSluice": -.46,
    # "immedAfterCataphoricSluice": 2.13,
    # "afterInitialSluice": -1.85,
    # "sluiceInCataphoricPattern": -1.03,
    # "LocativeCorr": 0,
    # "EntityCorr": 0,
    # "TemporalCorr": 0,
    # "DegreeCorr": 0,
    # "WhichCorr": 0
}

# subset of lambdas participating in hill climbing
ChangeLambdas = {
    # "lmScore": 10,
    # "ngram2": 10,
    # "ngram3": 10,
    # "w2vMaxSimilarity": -0.5,
    # "w2vMainPredicate": 2,
    "w2vPredicates": 2,
    # "distanceFromSluice": -1,
    # "sluiceCandidateOverlap": 1, 
    # "backwards": -1,
    # "WH_gov_npmi": [getFirst,0.5], #[first] 
    # "containsSluice": -10,
    # "isDominatedBySluice": -10,
    # "isInRelClause": -10,
    # "isInParenthetical": -10,
    # "coordWithSluice": 0,
    # "immedAfterCataphoricSluice": 10,
    # "afterInitialSluice": -10,
    # "sluiceInCataphoricPattern": 0,
    # "LocativeCorr": 0,
    # "EntityCorr": 0,
    # "TemporalCorr": 0,
    # "DegreeCorr": 0,
    # "WhichCorr": 0
}




correlatesByCand = defaultdict(lambda: defaultdict(list))
isInitial = True



# normalise the feature values if needed
def normaliseFeatures(features):
    sluiceCandidateOverlapMin = sluiceCandidateOverlapMax = 0.0
    npmiMin = npmiMax = 0.0
    lmMin = lmMax = 0.0

    # find min and max for each feature
    for sluiceId, candidates in features.items():
        for candidate in candidates:
            if npmiMin == 0.0 and npmiMax == 0.0:
                npmiMin =  npmiMax = candidate["WH_gov_npmi"][0]
            else:
                if candidate["WH_gov_npmi"][0] > npmiMax:
                    npmiMax = candidate["WH_gov_npmi"][0]
                if candidate["WH_gov_npmi"][0] < npmiMin and candidate["WH_gov_npmi"][0] > 0: 
                    npmiMin = candidate["WH_gov_npmi"][0]
                if candidate["sluiceCandidateOverlap"] > sluiceCandidateOverlapMax:
                    sluiceCandidateOverlapMax =  candidate["sluiceCandidateOverlap"]
                if candidate["sluiceCandidateOverlap"] < sluiceCandidateOverlapMin:
                    sluiceCandidateOverlapMin =  candidate["sluiceCandidateOverlap"]
                if "lmScore" in candidate:
                    if candidate["lmScore"] > lmMax:
                        lmMax = candidate["lmScore"]
                    if candidate["lmScore"] < lmMin:
                        lmMin = candidate["lmScore"]
                
    # normalise features
    for sluiceId, candidates in features.items():
        for candidate in candidates:
            if candidate["WH_gov_npmi"][1] == 'MISSING': #nothing left here, set it to the minimum
                candidate["WH_gov_npmi"][0] = 0
            else:
                candidate["WH_gov_npmi"][0] = (candidate["WH_gov_npmi"][0] - npmiMin) / (npmiMax - npmiMin)

            if candidate["WH_gov_npmi"][0] <0:
                pdb.set_trace()
            if candidate["sluiceCandidateOverlap"] > 0:
                candidate["sluiceCandidateOverlap"] = float(candidate["sluiceCandidateOverlap"] - sluiceCandidateOverlapMin) / float(sluiceCandidateOverlapMax - sluiceCandidateOverlapMin);
            if "lmScore" in candidate:
                candidate["lmScore"] = float(candidate["lmScore"] - lmMin) / float(lmMax - lmMin);

# modify certain features 
def modifyFeatures(features):
    for sluiceId, candidates in features.items():
        for candidate in candidates:
            # set backwards
            candidate["backwards"] = 1 if candidate["distanceFromSluice"] >= 0 else 0
                
            candidate["immedAfterCataphoricSluice"] = 0
            candidate["afterInitialSluice"] = 0
            candidate["sluiceInCataphoricPattern"] = 0
                
            if "do n't know why , but" in candidate["sluiceLineText"] or (candidate["sluiceInBut"] and candidate["sluiceNegated"]):
                candidate["sluiceInCataphoricPattern"] = 1
                if candidate["distanceFromSluice"] in [1,2] and candidate["backwards"] == 1:
                    candidate["immedAfterCataphoricSluice"] = 1
                        
            if candidate["sluiceIsInitial"] and not candidate["sluiceInCataphoricPattern"] and candidate["sentence"] == 1:
                candidate["afterInitialSluice"] = 1
                            
            if candidate["distanceFromSluice"] < 0:
                candidate["distanceFromSluice"] *= -1


def handleCorrFeat(sluiceType,key,corrVals, candidate):
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
        
def score(candidate, lambdas):
    score = 0
    sluiceType = candidate["sluiceType"]
    correlations = candidate["corrEls"]
    
    for key, factor in lambdas.items():
        if "Corr" in key:
            weight = handleCorrFeat(sluiceType, key, correlations, candidate)
            if isInitial:
                correlatesByCand[sluiceType][candidate["sluiceId"]].append(weight > 0)
        else:
            weight = candidate[key]               
            if type(weight) == type(True):
                weight = int(weight)
            elif type(factor) == type([]):
                weight = factor[0](weight)
                factor = factor[1]
            debug("scoring" + key + " " + str(weight) + " " + str(factor), 1)
        score += weight * factor
    return score


debugGlobal = 0
def debug(message, level):
    if debugGlobal >  level:
         print message

def computeBest(candidates, lambdas):    
    debug("Computing best for sluiceId: " + candidates[0]["sluiceId"], 1)
    for candidate in candidates:
        candidate["score"] = score(candidate, lambdas)
        debug("Scored " + str(candidate["score"]) + ": " + candidate["text"], 1)

    candidates = sorted(candidates, key=lambda x: x["score"])
    debug("Chose:" + candidates[-1]["text"], 1)

    return (candidates[-1], candidates)


def computeStats(chosen, correct):    
    try:
        chosenTokens = Counter(chosen["text"].split(" "))
        correctTokens = Counter(correct["text"].split(" "))
        totalChosenTokens = sum(chosenTokens.values())
        totalCorrectTokens = sum(correctTokens.values())
        
        deltaTokens = chosenTokens - correctTokens
        totalDeltaTokens = sum(deltaTokens.values())
        overlap = totalChosenTokens-totalDeltaTokens
    
        precision = float(overlap)/totalChosenTokens
        recall = float(overlap)/totalCorrectTokens
    except:
        pdb.set_trace()
        
    try:
        f = 2 * recall * precision / (precision + recall)
    except:
        f = 0.0

    return (precision, recall, f)


def predictAntecedent(features, lambdas):
    allScores = {}
    totalStats = {"p": 0.0, "r": 0.0, "f": 0.0, "total": 0}
    
    successesByType = defaultdict(lambda: defaultdict(int))
    failuresByType = defaultdict(lambda: defaultdict(int))
    statsBySentence = []
    for x in range(0,2):
        d = {"right": 0, "wrong": 0, "sameSentence": 0, "onlyoneCandSet": 0, "onlyoneCandSetInCorrectSent": 0, "overlapDiff": 0}
        statsBySentence.append(d)
    
    for sluiceId, candidates in features.items():
        correct = getAntecedent(candidates)
        sluiceType = correct["sluiceType"]
        correlateType = correct["corrType"]
        correctSentence = correct["sentence"]

        try:
            chosen, candidates = computeBest(candidates, lambdas)

            if  (chosen["sluiceCandidateOverlap"] - correct["sluiceCandidateOverlap"]) < 0:
                statsBySentence[correctSentence]["overlapDiff"] += 1
            if len(candidates) == 1:
                statsBySentence[correctSentence]["onlyoneCandSet"] += 1
            if len(filter(lambda x: x["sentence"] == correctSentence, candidates)) == 1:
                statsBySentence[correctSentence]["onlyoneCandSetInCorrectSent"] += 1
            if chosen["sentence"] == correctSentence:
                statsBySentence[correctSentence]["sameSentence"] += 1
            if chosen["isAntecedent"]:
                statsBySentence[correctSentence]["right"] += 1
                successesByType[sluiceType][correlateType] +=1
                debug("---------CORRECT----------", 1)
                debug("Id " + sluiceId + ": " + chosen["text"] + " " + str(chosen["distanceFromSluice"]) + " " + str(chosen["backwards"]), 1)
                debug("--------------------------", 1)
            else:
                statsBySentence[correctSentence]["wrong"] += 1
                failuresByType[sluiceType][correlateType] +=1
                debug("---------WRONG----------", 1)
                debug("Id " + sluiceId + ": " + chosen["text"], 1)
                debug("Antecedent: " + correct["text"], 1)
                debug("Sluice: " + chosen["sluiceLineText"] + " " + str(chosen["distanceFromSluice"]) + " " + str(chosen["backwards"]), 1)
                debug("--------------------------", 1)
                
            #p,r,f = computeStats(chosen,correct)
            p,r,f = chosen["antePRF"]
            totalStats["p"] += p
            totalStats["r"] += r
            totalStats["f"] += f
            totalStats["total"] += 1
            
        except Exception, e:
            print "Error in antecedent prediction computation for " + sluiceId
            traceback.print_exc(file=sys.stdout)
            continue

    for x in range(0,2):
        statsBySentence[x]["total"] = statsBySentence[x]["wrong"] + statsBySentence[x]["right"]
    
    allScores["prevSame"] = {"prev": statsBySentence[0], "same": statsBySentence[1]}
    allScores["p"] = totalStats["p"]/totalStats["total"]
    allScores["r"] = totalStats["r"]/totalStats["total"]
    allScores["f"] = totalStats["f"]/totalStats["total"]
    allScores["accuracy"] = float(allScores["prevSame"]["prev"]["right"] + allScores["prevSame"]["same"]["right"]) / (allScores["prevSame"]["prev"]["total"] + allScores["prevSame"]["same"]["total"])
    allScores["statsByType"] = (successesByType, failuresByType)
    return allScores

def randomStart():
    l = Lambdas
    l2 = copy.deepcopy(l)
    random.seed()
    for x in l.keys():
    #    v = random.random()*pow(10,random.randint(0,2))
        v = random.randint(-5,5)
        if type(l[x]) == type([]):
            l2[x] = [l[x][0], v]
        else:
            l2[x] = v
    return l2         

def zeroStart():
    l = Lambdas
    l2 = copy.deepcopy(l)
    for x in l.keys():
        v = 0
        if type(l[x]) == type([]):
            l2[x] = [l[x][0], v]
        else:
            l2[x] = v
    return l2         


def optimize(features, iterations = 100, restarts=10, maxIncr=30):
    
    def makeLambdaField(l, incr=1):
        field = [l]
#        for x in l.keys():
        for x in ChangeLambdas.keys():
            dplus = copy.deepcopy(l)
            dminus = copy.deepcopy(l)
            if type(l[x]) == type([]):
                dplus[x] = [dplus[x][0], l[x][1]+incr]
                dminus[x] = [dminus[x][0], l[x][1]-incr]
            else:
                dplus[x] = l[x]+incr
                dminus[x] = l[x]-incr
            field.append(dplus)
            field.append(dminus)
        return field
    
    def getF(scoreStr):
        return scoreStr["f"]

    def getAcc(scoreStr):
        return scoreStr["accuracy"]
        
#    lMax = Lambdas

#    lMax = zeroStart()

    maxScore = None
    incr = float(maxIncr)
    restartMax = 0
    restartMaxScoreStr = ""
    restartMaxScore = maxScore
    #iterations = 100
    
    for r in range(0, restarts):
        lMax = randomStart()
        maxScore = 0
        print "Restart", r
        prevMax = 0
        for it in range(iterations):
            print restarts, iterations
            print "Iteration", it
            field = makeLambdaField(lMax, incr = maxIncr/math.sqrt(float(it+1)))
            for l in field:
                scoreStr = predictAntecedent(features,l)
                curScore = getAcc(scoreStr)
                if debugGlobal > 0:
                    fldValsPrint(l),
                    print curScore

                if curScore > maxScore:
                    maxScore= curScore
                    maxScoreStr = scoreStr
                    lMax = l
                    
            if prevMax == maxScore:
                break
            else:
                prevMax = maxScore

            print "Lmax:", 
            fldValsPrint(lMax)
            print "MaxScore:",maxScore
                    

        if maxScore > restartMaxScore:
            restartMaxScore = maxScore
            restartMax = lMax
            restartMaxScoreStr = maxScoreStr
                   
    return restartMax, restartMaxScoreStr

def optimize_org(features, iterations = 10, maxIncr=30):
    
    def makeLambdaField(l, incr=1):
        field = [l]
#       for x in l.keys():
        for x in ChangeLambdas.keys():

            dplus = copy.deepcopy(l)
            dminus = copy.deepcopy(l)
            if type(l[x]) == type([]):
                dplus[x] = [dplus[x][0], l[x][1]+incr]
                dminus[x] = [dminus[x][0], l[x][1]-incr]
            else:
                dplus[x] = l[x]+incr
                dminus[x] = l[x]-incr
            field.append(dplus)
            field.append(dminus)
        return field
    
    def getF(scoreStr):
        return scoreStr["f"]

    def getAcc(scoreStr):
        return scoreStr["accuracy"]
        
    lMax = Lambdas
#   lMax = randomStart()
#   lMax = zeroStart()

    maxScore = None
    for it in range(iterations):
        print "Iteration", it
        field = makeLambdaField(lMax, incr = maxIncr/float(it+1))
        print "Lmax:", 
        lambdaPrint(lMax)
        print "MaxScore:",maxScore
        for l in field:
            scoreStr = predictAntecedent(features,l)
            curScore = getAcc(scoreStr)
            if debugGlobal > 0:
                print "Lambdas:", 
                lambdaPrint(l)
                print curScore

            if maxScore == None or curScore > maxScore:
                maxScore= curScore
                maxScoreStr = scoreStr
                lMax = l
    return lMax, maxScoreStr

def printNumCorrs():
    sluiceTypes = correlatesByCand.keys()
    for s in sluiceTypes:
        sluices = correlatesByCand[s]
        totSluices = float(len(sluices))
        propTotal = 0
        for id,sl in sluices.items():
            totCands = float(len(sl))
            contain = len([x for x in sl if x == True])
            proportion = contain/totCands
            propTotal += proportion
        print propTotal, totSluices
        print "%s\t%f" % (s, propTotal/totSluices)

def printSF(sucF):
    suc, fail = sucF
    sluiceTypes = set(fail.keys()).union(set(suc.keys()))
    corrT = [set(x.keys()) for x in suc.values()]
    corrT.extend([set(x.keys()) for x in fail.values()])
    corrTypes = reduce(lambda x,y: x.union(y), corrT)

    for s in sluiceTypes:
        for c in corrTypes:
            if suc[s][c] == fail[s][c] == 0:
                continue
            print '\t'.join([str(s), str(c), str(suc[s][c]), str(fail[s][c])])

    # for s,h in suc.items():
    #     for c in h.keys():
    #         print '\t'.join([str(s), str(c), str(h[c])])
    # 
    # print
    # print
    # 
    # for s,h in fail.items():
    #     for c in h.keys():
    #         print '\t'.join([str(s), str(c), str(h[c])])


def lambdaPrint(lambdas):
    for k in sorted(lambdas.keys()):
        if type(lambdas[k]) == type([]):
            debug("\t\t" + k + ":\t" + str(lambdas[k][1]), 0)
        else:
            debug("\t\t" + k + ":\t" + str(lambdas[k]), 0)

def fldValsPrint(l):
    for k in sorted(l.keys()):
        if type(l[k]) == type([]):
            print l[k][1],
        else:
            print l[k],
    print 

def delta(d1, d2):
    for k in d1.keys():
        if type(d1[k]) == type([]):
            d1[k][1] -= d2[k][1]
            if d1[k][1] == 0:
                del d1[k]
        else:
            d1[k] -= d2[k]
            if d1[k] == 0:
                del d1[k]
    return d1
        


####### ---->
###         ------------>
###     LET'S GO        ----------->
###         ------------>
####### ---->
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Take file of features (as jsons) as compute antecedent.')
    parser.add_argument('-debug', metavar='debug', type=int, help='debug level')
    parser.add_argument('-optimize', metavar='optimize', type=int, help='optimize switch')
    parser.add_argument('-restarts', metavar='restarts', type=int, help='number of random restarts')
    parser.add_argument('-iterations', metavar='iterations', type=int, help='number of iterations in hill-climbing')
    parser.add_argument('featuref', metavar='featuref', type=str, help='the featurefile')
                       
    args = parser.parse_args()
    features = loadData(args.featuref)
       
    debugGlobal = 0
    if args.debug:
        debugGlobal = args.debug

    optimize_switch = 1
    if args.optimize:
        optimize_switch = 1

    restarts = 1
    if args.restarts:
        restarts = args.restarts
    
    iterations = 100
    if args.iterations:
        iterations = args.iterations

    debug("Starting Lambdas: ", 0)
    lambdaPrint(Lambdas)
    debug("--------------------", 0)

    # preprocess features
    normaliseFeatures(features)
    modifyFeatures(features)

    currentScore = predictAntecedent(features, Lambdas)
    isInitial = False
    print "Initial Score: ", currentScore

    sF = currentScore["statsByType"]
    
    printSF(sF)
    printNumCorrs()

    if optimize_switch:
        lambdas, scoreStr = optimize(features, iterations=iterations,restarts=restarts)
        print delta(lambdas, Lambdas)
        print "Score: ", scoreStr







