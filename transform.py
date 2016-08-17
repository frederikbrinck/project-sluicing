import kenlm
import json
import collections
from nltk import pos_tag, word_tokenize
from gensim.models import Word2Vec as w2v

from lib.data import loadData, saveData, tableFromData
from lib.functions import getSluice
from lib.probability import computeProbability
from features.f_language import extractFeatures

if __name__ == '__main__':
	import argparse

	# config
	#	"lm": on/off
	#	"pos": [leastNgram, highestNgram]
	# 	"w2v": on/off
	config = { "lm": 0, "pos": 0, "w2v": 1 }

	# setup parser and parse args
	parser = argparse.ArgumentParser(description='Transforms the sluice example data adding or removing features')
	parser.add_argument('dataref', metavar='dataref', type=str, help='Reference to the example file')
	parser.add_argument('--save', metavar='save', type=str, help='Destination for the file save')
	args = parser.parse_args()

	# load data 
	examples = loadData(args.dataref)

	# ---------------------------------------------------------------
	# add language model specs to examples
	# ---------------------------------------------------------------
	if config["lm"] != 0: 
		# load language model
		lmModel = kenlm.Model('models/standard.arpa')

		# run through all examples
		for sluiceId in examples:
			for sentence in examples[sluiceId]:
					# form sentence to pitch against
					# the language model 
					candidate = sentence["text"]
					sluice = sentence["sluiceGovVPText"]
					pitch = sluice + " " + candidate;
					pitch = pitch.split(" ");
					pitch = " ".join(pitch[0:len(sluice.split(" ")) + min(5, len(candidate.split(" ")))])

					# calculate the probability for the 
					# pitch sentence usin the KenLM.
					# Consider switiching to a better
					# language model.
					sentence["lmScore"] = lmModel.score(pitch.lower(), bos=False, eos=False)

		# write stats to file
		with open("stats/lm-oov-counts.stat", "w") as fp:
			oovs = collections.Counter()
			for sluiceId in examples:
				sentenceNum = 0
				for sentence in examples[sluiceId]:
					words = word_tokenize(sentence["text"])
					for word in words[0:min(5, len(words))]:
						if word not in lmModel:
							if word not in oovs:
								oovs[word] = 1
							else:
								oovs[word] += 1

			for common in oovs.most_common():
				line = common[0] + ", " + str(common[1]) + "\n"
				fp.write(line)


					

	# ---------------------------------------------------------------
	# add pos-tagging specs to examples
	# ---------------------------------------------------------------
	if config["pos"] != 0:
		# get probabilities and run over example
		probabilities = tableFromData(examples, config["pos"][0], config["pos"][1])
		for sluiceId in examples:
			for sentence in examples[sluiceId]:
				# extract sluice and pos tags
				# from candidate
				candidate = sentence["text"]
				tags = [m[1] for m in pos_tag(candidate)]
				sluice = getSluice(sentence["sluiceGovVPText"])
				if not sluice:
					sluice = sentence["sluiceGovVPText"]

				# calculate probabilities
				for i in range(config["pos"][0], config["pos"][1] + 1):
					sentence["ngram" + str(i)] = computeProbability(i, tags, sluice, probabilities)


	# ---------------------------------------------------------------
	# add word2vec specs to examples
	# ---------------------------------------------------------------
	if config["w2v"] != 0:
		# get the sluice similarity by breaking up the sluice
		# into words and taking the max
		def getSluiceSimilarity(word, sluice):
			similarity = -1
			word = [x.lower() for x in word]
			sluice = [x.lower() for x in sluice.split(" ")]
			try: 
				similarity = model.n_similarity(sluice, word)
			except: 
				return similarity
			
			return similarity

		def getVerbs(words):
			return [word[0] for word in words if word[1] == "VERB"]

		# load model, this might take some time
		model = w2v.load_word2vec_format("models/GoogleNews-vectors-negative300.bin", binary=True)
		# model = w2v.load("models/text8.model")

		# run over each example
		wordStats = {}
		for sluiceId in examples:
			sentenceNum = 0
			wordStats[sluiceId] = [None for x in range(len(examples[sluiceId]))]
			for sentence in examples[sluiceId]:
				# get similarity in case we don't 
				# have containment
				wordStats[sluiceId][sentenceNum] = {}
				wordStats[sluiceId][sentenceNum]["similarities"] = []
				if sentence["containsSluice"]:
					# add features
					sentence["w2vMaxSimilarity"] = -1
					sentence["w2vMainPrediate"] = -1
					sentence["w2vPredicates"] = -1
					# set stats
					wordStats[sluiceId][sentenceNum]["similarities"].append([-1, "Containment"])
					wordStats[sluiceId][sentenceNum]["mainPredicate"] = [-1, sentence["WH_gov_npmi"][1]]
					wordStats[sluiceId][sentenceNum]["predicates"] = [-1, ["Containment"]]
				else:
					# add features
					maxSimilarity = count = 0
					words = word_tokenize(sentence["text"])
					verbs = getVerbs(pos_tag(words, tagset='universal'))
					for word in words:
						if word in getSluice(sentence["sluiceGovVPText"]):
							continue

						similarity = getSluiceSimilarity(word, getSluice(sentence["sluiceGovVPText"]))
						maxSimilarity = max(similarity, maxSimilarity);
						wordStats[sluiceId][sentenceNum]["similarities"].append([similarity, word])

					sentence["w2vMaxSimilarity"] = maxSimilarity
					sentence["w2vMainPredicate"] = getSluiceSimilarity(sentence["WH_gov_npmi"][1], getSluice(sentence["sluiceGovVPText"]))
					sentence["w2vPredicates"] = getSluiceSimilarity(verbs, getSluice(sentence["sluiceGovVPText"])) if len(verbs) > 0 else -1
					# set stats
					wordStats[sluiceId][sentenceNum]["mainPredicate"] = [getSluiceSimilarity(sentence["WH_gov_npmi"][1], getSluice(sentence["sluiceGovVPText"])), sentence["WH_gov_npmi"][1]]
					wordStats[sluiceId][sentenceNum]["predicates"] = [getSluiceSimilarity(verbs, getSluice(sentence["sluiceGovVPText"])), verbs] if len(verbs) > 0 else [-1, ["None"]]
					wordStats[sluiceId][sentenceNum]["similarities"] = sorted(wordStats[sluiceId][sentenceNum]["similarities"], reverse=True)
				sentenceNum += 1
		
		# write stats to file
		with open("stats/w2v-similarity-sentences.stat", "w") as fp:
			for sluiceId in examples:
				sentenceNum = 0
				fp.write("================\n\n")
				for sentence in examples[sluiceId]:
					line = "Sentence: " + sentence["text"] + "\n"
					line += "Sluice: " + sentence["sluiceGovVPText"] + "\n"
					line += "Antecedent: " + str(sentence["isAntecedent"]) + "\t\tMain Predicate: " + str(wordStats[sluiceId][sentenceNum]["mainPredicate"][0]) + ", " + wordStats[sluiceId][sentenceNum]["mainPredicate"][1] + "\t\tPredicates: " + str(wordStats[sluiceId][sentenceNum]["predicates"][0]) + ", " + ", ".join(wordStats[sluiceId][sentenceNum]["predicates"][1]) + "\n"
					for stat in wordStats[sluiceId][sentenceNum]["similarities"][0:3]:
						line += "\t\t" + str(stat[0]) + ", " + str(stat[1]) + "\n"
					line +="\n"
					fp.write(line)
					sentenceNum += 1

		# write stats to file
		with open("stats/w2v-similarity-counts.stat", "w") as fp:
			tierOne = collections.Counter()
			for sluiceId in examples:
				sentenceNum = 0
				for sentence in examples[sluiceId]:
					if wordStats[sluiceId][sentenceNum]["similarities"][0][1] not in tierOne:
						tierOne[wordStats[sluiceId][sentenceNum]["similarities"][0][1]] = 1
					else:
						tierOne[wordStats[sluiceId][sentenceNum]["similarities"][0][1]] += 1
					sentenceNum += 1
			for common in tierOne.most_common():
				line = common[0] + ", " + str(common[1]) + "\n"
				fp.write(line)

	if args.save:
		saveData(args.save, examples, 1);