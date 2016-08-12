import kenlm
import json
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
					pitch = " ".join(pitch[0:min(9, len(pitch))])

					# calculate the probability for the 
					# pitch sentence usin the KenLM.
					# Consider switiching to a better
					# language model.
					sentence["lmScore"] = lmModel.score(pitch)

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
			try: 
				for sluicePart in word_tokenize(sluice):
					similarity = max(model.similarity(sluicePart, word), similarity)
			except:
				return similarity
			return similarity

		# load model, this might take some time
		model = w2v.load_word2vec_format("models/GoogleNews-vectors-negative300.bin", binary=True)
		# model = w2v.load("models/text8.model")

		# run over each example
		wordStats = {}
		for sluiceId in examples:
			sentenceNum = 0
			wordStats[sluiceId] = [None for x in range(len(examples[sluiceId]))]
			for sentence in examples[sluiceId]:
				# get similarity
				total = count = 0
				words = word_tokenize(sentence["text"])
				wordStats[sluiceId][sentenceNum] = []
				for word in words:
					if word in getSluice(sentence["sluiceGovVPText"]):
						continue

					similarity = getSluiceSimilarity(word, getSluice(sentence["sluiceGovVPText"]))
					total = max(similarity, total);
					wordStats[sluiceId][sentenceNum].append([similarity, word])

				wordStats[sluiceId][sentenceNum] = sorted(wordStats[sluiceId][sentenceNum], reverse=True)
				sentence["w2vMaxSimilarity"] = total
				sentenceNum += 1
		
		# write stats to file
		with open("stats/w2v-similarity", "w") as fp:
			for sluiceId in examples:
				sentenceNum = 0
				for sentence in examples[sluiceId]:
					line = "Sentence: " + sentence["text"] + "\n"
					line += "Sluice: " + sentence["sluiceGovVPText"] + "\n"
					line += "Antecedent: " + str(sentence["isAntecedent"]) + "\n"
					for stat in wordStats[sluiceId][sentenceNum][0:3]:
						line += "\t\t" + str(stat[0]) + ", " + str(stat[1]) + "\n"
					line +="\n"
					fp.write(line)
					sentenceNum += 1

	saveData(args.save, examples, 1);