# data.py holds functions for manipulating
# data specifically dealing with data i/o
# -----------------------------------------
import json
import probability
from nltk import pos_tag
from functions import getAntecedents, getNgrams, getSluice, getProbabilities
from probability import computeProbability
from collections import defaultdict

# load example data from the given file
# and index it based on the key 
def loadData(file, key = "sluiceId"):
    examples = defaultdict(list)

    # open and read file
    fd = open(file, "r")
    for line in fd:
        data = json.loads(line)
        try:
            sluiceId = data[key]
        except:
            continue
        examples[sluiceId].append(data)

    return examples



# save data to file in either a full
# or a compact format
def saveData(file, saveDict, compact = 0):
    with open(file, "w") as fp:
        # make sure to write each entry
        # on a separate line
        if (compact):
            for k in saveDict:
                for sentence in saveDict[k]:
                    sentence["sluiceId"] = k
                    fp.write(json.dumps(sentence) + "\n")
                # out = json.dumps(saveDict[k])
                # fp.write(out.strip('"') + "\n")
        else:
            json.dump(saveDict, fp, indent = 4)



# given a set of examples
# return the a dictionary of examples 
# from position start to position end;
# we can do this as long as the dictionary
# doesn't change in between
def splitData(examples, start, end):
    i = 0
    batch = {}
    for k in examples.keys():
        if i >= start and i < end:
            batch[k] = examples[k]
        i += 1

    return batch



# filter the data by separating out
# all data of the correct length
def filterData(examples, length=False):
    if not length:
        print "Error: No valid parameters have been given to filterData"
        return

    filteredData = {}
    if length:
        for k in examples.keys():
            if len(examples[k]) == length:
                filteredData[k] = examples[k]

    return filteredData




# create the probability table from
# the given data
def tableFromData(examples, ngramLow, ngramHigh):
    antecedents = getAntecedents(examples)
    ngrams = getNgrams(antecedents, ngramLow, ngramHigh + 1)
    probabilities = getProbabilities(ngrams)

    return probabilities



# given examples, a table of probabilities
# and other settings, predict the model
# accuracy on the data
def predictData(examples, table, ngramLow, ngramHigh, coefs, verbose = 1):
    correctlyPredicted = 0.0

    if sum(coefs) > 1.1 or sum(coefs) < 0.9:
        print "Coefficients don't sum to one"
        return

    # run over data and predict for each
    # example
    for sluiceId in examples:
        predictedProbability = 0.0
        predictedAntecedent = ""
        realAntecedent = ""

        for example in examples[sluiceId]:
            # get candidate and correct
            # antecedent
            candidate = example["text"]
            if example["isAntecedent"]:
                realAntecedent = candidate

            # extract sluice and pos tags
            tags = [m[1] for m in pos_tag(candidate)]
            sluice = getSluice(example["sluiceGovVPText"])
            if not sluice:
                sluice = example["sluiceGovVPText"]

            # calculate probabilities
            tempProbability = 0
            coef = 1.0 / (ngramHigh - ngramLow)
            for i in range(ngramLow, ngramHigh):
                tempProbability += coefs[i - ngramLow] * computeProbability(i, tags, sluice, table)

            if tempProbability > predictedProbability:
                predictedAntecedent = candidate
                predictedProbability = tempProbability

        # count number of correct predictions
        # and print out information
        if predictedAntecedent == realAntecedent:
            correctlyPredicted += 1.0

        if verbose:
            print "-------"
            print "Predicted: (", predictedAntecedent == realAntecedent, "): ", predictedProbability
            print "Candidate: ", predictedAntecedent
            print "Actual: ", realAntecedent  
            print "-------"

    return correctlyPredicted / len(examples), correctlyPredicted, len(examples) - correctlyPredicted