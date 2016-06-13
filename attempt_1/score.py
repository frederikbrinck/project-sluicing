#!/usr/bin/python
import csv
import json
from collections import defaultdict, Counter
import pdb
import sys, traceback
import copy
import random
import math

def getFirst(x):
    return x[0]

correlatesByCand = defaultdict(lambda: defaultdict(list))
isInitial = True
# Lambdas = {"sluiceInParenthetical": 1,
# "sentence": -1, 
# "sluiceInBut": 1,
# "sluiceCandidateOverlap": 1, 
# "sluiceNegated": 1, 
# "distanceFromSluice": -1, 
# "sluiceInCoordinatedVP": 1,
# "WH_gov_npmi": [getFirst,pow(10,9)], #[first]
# "containsSluice": -1,
# "isDominatedBySluice": -1,
# "WH_gov_count": [getFirst,-.05],  #[first]
# "embedsS": -1, 
# #"containsAntecedent": -1,
# "isInRelClause": -1, 
# "isInParenthetical": -1, 
# "coordWithSluice": 1
# }

# Lambdas = {
#     # "sluiceInParenthetical": 0,
#     # "sentence": 0,
#     # "sluiceInBut": 0,
#     # "sluiceIsInitial": 0,
#     # "sluiceCandidateOverlap": 0,
#     "sluiceNegated": 1,
#      "distanceFromSluice": -1,
#     "backwards": -1,
# #    "sluiceInCoordinatedVP": 0,
#     "WH_gov_npmi": [getFirst,0], #[first] 
#     "containsSluice": -10,
#     "isDominatedBySluice": -10,
#     "WH_gov_count": [getFirst,0], #[first]
# #    "embedsS": 0,
# #   "containsAntecedent": -1,
#     "isInRelClause": -10,
#     "isInParenthetical": -10,
#     "coordWithSluice": 0,
#     "immedAfterCataphoricSluice": 10,
#     "afterInitialSluice": -10,
# #    "largerThanEquivCand": 0
#      }

# Lambdas = {
#     "distanceFromSluice": -1,
#     "sluiceCandidateOverlap": 1, 
#     "backwards": -1,
#     "WH_gov_npmi": [getFirst,0], #[first] 
#     "containsSluice": -10,
#     "isDominatedBySluice": -10,
#     "isInRelClause": -10,
#     "isInParenthetical": -10,
#     "coordWithSluice": 0,
#     "immedAfterCataphoricSluice": 10,
#     "afterInitialSluice": -10,
#     "sluiceInCataphoricPattern": 0,
#     "LocativeCorr": 1,
#     "EntityCorr": 0,
#     "TemporalCorr": 1,
#     "DegreeCorr": 0,
#     "WhichCorr": 1
    
#      }


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



# this could define a subset of Lambdas that should participate in Hill Climbing
ChangeLambdas = {
    "distanceFromSluice": -1,
#    "sluiceCandidateOverlap": 1, 
    "backwards": -1,
#   "WH_gov_npmi": [getFirst,0.5], #[first] 
    "containsSluice": -10,
    "isDominatedBySluice": -10,
    "isInRelClause": -10,
    "isInParenthetical": -10,
    "coordWithSluice": 0,
    "immedAfterCataphoricSluice": 10,
    "afterInitialSluice": -10,
    "sluiceInCataphoricPattern": 0,
    "LocativeCorr": 0,
    "EntityCorr": 0,
    "TemporalCorr": 0,
    "DegreeCorr": 0,
    "WhichCorr": 0
}


def loadData(fname):
    features = defaultdict(list)
    fd = open(fname, "r")
    #csv.reader(fd)
    for row in fd:
        d = json.loads(row)
        try:
            sluiceId = d["sluiceId"]
        except:
            continue
        features[sluiceId].append(d)

    return features


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




def getCorrect(candidateSet):
    correct = None
    for c in candidateSet:
        if c["isAntecedent"]:
            return c
    
    return None

def computeBest(candidateSet, lambdas):
    
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
        
    def score(cand):
        s = 0
        sluiceType = cand["sluiceType"]
        corrVals = cand["corrEls"]
        
        for key,factor in lambdas.items():
            if "Corr" in key:
                c= handleCorrFeat(sluiceType, key, corrVals, cand)
                if isInitial:
                    correlatesByCand[sluiceType][cand["sluiceId"]].append(c > 0)
            else:
                c = cand[key]               
                if type(c) == type(True):
                    c = int(c)
                elif type(factor) == type([]):
                    c = factor[0](c)
                    factor = factor[1]
                if debug_level > 1:
                    print "scoring", key, c, factor
            s += c * factor
        return s


    if debug_level > 1:
        print "SLUICEID: ", candidateSet[0]["sluiceId"]
    for c in candidateSet:
        c["score"] = score(c)
        if debug_level > 1:
            print c["text"]
            print c["score"]

    cands = sorted(candidateSet, key=lambda x: x["score"])
    if debug_level > 1:
        print "chosen", cands[-1]["text"]

    return (cands[-1], cands)

def computePrecRec(chosen,correct):
    
    def getToks(s):
        c = Counter(s.split(" "))
        return c
    try:
        ChToks = getToks(chosen["text"])
        CoToks = getToks(correct["text"])
        DiffToks = ChToks-CoToks
        totChToks = sum(ChToks.values())
        totCoToks = sum(CoToks.values())
        totDiffToks = sum(DiffToks.values())
        totInToks = totChToks-totDiffToks
    
        p = float(totInToks)/totChToks
        r = float(totInToks)/totCoToks
    except:
        pdb.set_trace()
        
    try:
        f = 2*r*p/(p+r)
    except:
        f = 0.0
    return (p,r,f)
    
def predictAntecedent(features, lambdas):
    allScores = {}
    successAndFailure = []
    totPrecRec = {"p": 0.0, "r": 0.0, "f": 0.0, "tot": 0}
    successesByType = defaultdict(lambda: defaultdict(int))
    failuresByType = defaultdict(lambda: defaultdict(int))
    
    for x in range(0,2):
        d = {"right": 0, "wrong": 0, "sameSentence": 0, "onlyoneCandSet": 0, "onlyoneCandSetInCorrectSent": 0, "overlapDiff": 0}
        successAndFailure.append(d)
    
    for sluiceId, candidateSet in features.items():
        correct = getCorrect(candidateSet)
        sluiceType = correct["sluiceType"]
        corrType = correct["corrType"]
        correctSentence = correct["sentence"]

        try:
            overlapUsed = False
            chosen, candSet = computeBest(candidateSet, lambdas)

            if  (chosen["sluiceCandidateOverlap"] - correct["sluiceCandidateOverlap"]) < 0:
                successAndFailure[correctSentence]["overlapDiff"] += 1
            if len(candSet) == 1:
                successAndFailure[correctSentence]["onlyoneCandSet"] += 1
            
            if len(filter(lambda x: x["sentence"] == correctSentence, candSet)) == 1:
                successAndFailure[correctSentence]["onlyoneCandSetInCorrectSent"] += 1
                
            if chosen["sentence"] == correctSentence:
                successAndFailure[correctSentence]["sameSentence"] += 1
                
                
            if chosen["isAntecedent"]:
                successAndFailure[correctSentence]["right"] += 1
                successesByType[sluiceType][corrType] +=1
                
                if debug_level > 1:
                    print "***Correct", sluiceId, chosen["text"], chosen["distanceFromSluice"], chosen["backwards"]
            else:
                successAndFailure[correctSentence]["wrong"] += 1
                failuresByType[sluiceType][corrType] +=1
                if debug_level > 1:
                    print "***Wrong", sluiceId, chosen["text"], "||", correct["text"], "||", chosen["sluiceLineText"],chosen["distanceFromSluice"], chosen["backwards"]
            #p,r,f = computePrecRec(chosen,correct)
            p,r,f = chosen["antePRF"]
            totPrecRec["p"] += p
            totPrecRec["r"] += r
            totPrecRec["f"] += f
            totPrecRec["tot"] += 1
            
        except Exception, e:
            print >> sys.stderr,"ERROR!!"
            traceback.print_exc(file=sys.stderr)
            continue
    for x in range(0,2):
        successAndFailure[x]["total"] = successAndFailure[x]["wrong"] + successAndFailure[x]["right"]
    allScores["prevSame"] = {"prev": successAndFailure[0], "same": successAndFailure[1]}
    allScores["p"] = totPrecRec["p"]/totPrecRec["tot"]
    allScores["r"] = totPrecRec["r"]/totPrecRec["tot"]
    allScores["f"] = totPrecRec["f"]/totPrecRec["tot"]
    allScores["acc"] = float(allScores["prevSame"]["prev"]["right"]+allScores["prevSame"]["same"]["right"])/(allScores["prevSame"]["prev"]["total"]+allScores["prevSame"]["same"]["total"])
    allScores["sucFail"] = (successesByType, failuresByType)
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


def optimize(features, iterations = 100, restarts=10, maxIncr=10):
    
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
        return scoreStr["acc"]
        
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
                if debug_level > 0:
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

def optimize_org(features, iterations = 10, maxIncr=20):
    
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
        return scoreStr["acc"]
        
    lMax = Lambdas
#   lMax = randomStart()
#   lMax = zeroStart()

    maxScore = None
    for it in range(iterations):
        print "Iteration", it
        field = makeLambdaField(lMax, incr = maxIncr/float(it+1))
        print "Lmax:", 
        fldPrint(lMax)
        print "MaxScore:",maxScore
        for l in field:
            scoreStr = predictAntecedent(features,l)
            curScore = getAcc(scoreStr)
            if debug_level > 0:
                print "Lambdas:", 
                fldPrint(l)
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


def fldPrint(l):
    for k in sorted(l.keys()):
        if type(l[k]) == type([]):
            print k, ": ", l[k][1],
        else:
            print k, ": ", l[k],
    print 

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
        
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Take file of features (as jsons) as compute antecedent.')
    parser.add_argument('-debug', metavar='debug', type=int, 
                       help='debug level')

    parser.add_argument('-optimize', metavar='optimize', type=int, 
                       help='optimize switch')

    parser.add_argument('-restarts', metavar='restarts', type=int, 
                       help='number of random restarts')

    parser.add_argument('-iterations', metavar='iterations', type=int, 
                       help='number of iterations in hill-climbing')

    parser.add_argument('featuref', metavar='featuref', type=str, 
                       help='the featurefile')
                       
    args = parser.parse_args()
    optimize_switch = 1
    debug_level = 0
    features = loadData(args.featuref)

    if args.debug:
        debug_level = args.debug

    if args.optimize:
        optimize_switch = 1

    if args.restarts:
        restarts = args.restarts
    else:
        restarts = 1

    if args.iterations:
        iterations = args.iterations
    else:
        iterations = 100

    print "starting lambdas:"
    fldPrint(Lambdas)

    normalizeVals(features)
    modifyFeatures(features)

    scoreStr = predictAntecedent(features,Lambdas)
    isInitial = False
    print "Initial Score: ",scoreStr

    sF = scoreStr["sucFail"]
    
    printSF(sF)
    printNumCorrs()

    if optimize_switch:
        lambdas, scoreStr = optimize(features, iterations=iterations,restarts=restarts)
        print delta(lambdas, Lambdas)
        print "Score: ", scoreStr






