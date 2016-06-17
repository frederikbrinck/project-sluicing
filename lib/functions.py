# functions.py contains standard functions
# used in conjunction with nltk to 
# calculate statistics
# -----------------------------------------
import re
import nltk
from nltk import pos_tag



# return the sluice of the given
# string, if any
def getSluice(haystack):
	sluiceRe = r'(how far|how long|how many|how much|how come|how old|when|where|why|how|what|which|who|whom|whose)'
	search = re.search(sluiceRe, haystack, re.I)
	if search:
		return search.group(1)
	else:
		return None



# extracts the sluiceId, the sluice phrase,
# the full phrase, and the antecedent itself
# from a given set of examples
def getAntecedents(examples):
	antecedentDictionary = {}
	for k in examples:
		antecedentDictionary[k] = {}

		# find the antecedent tree
		for i in range(len(examples[k])):
			cEx = examples[k][i]
			if (cEx["isAntecedent"]): 
				antecedentDictionary[k]["tags"] = pos_tag(cEx["text"])
				if (getSluice(cEx["sluiceGovVPText"]) == None):
					antecedentDictionary[k]["sluice"] = cEx["sluiceGovVPText"]
				else: 
					antecedentDictionary[k]["sluice"] = getSluice(cEx["sluiceGovVPText"])
	
	return antecedentDictionary



# get ngrams of all antecedents requiring
# that the antecedents is a dictionary
# containing examples with a tag list and
# a sluice
def getNgrams(antecedents, least, highest):
	# create list and iterate 
	# over tags
	ngramlist = {}
	for key in antecedents:
		# get pos tags and sluice
		tags = [m[1] for m in antecedents[key]["tags"]]
		sluice = antecedents[key]["sluice"]

		# make ngrams and add them to 
		# the dictionary as lists categorised
		# by their sluices
		for i in range(least, highest):
			if str(i) not in ngramlist:
				ngramlist[str(i)] = {}

			if sluice not in ngramlist[str(i)]:
				ngramlist[str(i)][sluice] = []

			for gram in list(nltk.ngrams(tags, i)):
				ngramlist[str(i)][sluice].append(gram)

	return ngramlist



# get the probabilities of the occurences
# of all the ngrams given a dictionary
# with the format 
# "1": { "sluice": [tags] }
def getProbabilities(ngrams):
	# calculate counts grouped by the
	# different sluices
	counts = {}
	for ngram in ngrams:
		counts[ngram] = {}

		for sluice in ngrams[ngram]:
			if sluice not in counts[ngram]:
				counts[ngram][sluice] = {}
				counts[ngram][sluice]["length"] = 0

			# count the tags for the current sluice
			for tags in ngrams[ngram][sluice]:
				entry = " ".join(tags)

				if entry not in counts[ngram][sluice]:
					counts[ngram][sluice][entry] = 1
				else:
					counts[ngram][sluice][entry] += 1

				counts[ngram][sluice]["length"] += 1

	# calculate probabilities of the tags
	# grouped by the sluices
	probabilities = {}
	for ngram in ngrams:
		probabilities[ngram] = {}
		for sluice in counts[ngram]:
			if sluice not in probabilities[ngram]:
				probabilities[ngram][sluice] = {}

			for entry in counts[ngram][sluice]:
				if entry != "length":
					probabilities[ngram][sluice][entry] = float(counts[ngram][sluice][entry]) / counts[ngram][sluice]["length"]

	return probabilities