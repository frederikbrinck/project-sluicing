import kenlm
import json
from nltk import pos_tag

from lib.data import loadData, saveData, tableFromData
from lib.functions import getSluice
from lib.probability import computeProbability
from features.f_language import extractFeatures

if __name__ == '__main__':
	import argparse

	# config
	#	"lm": on/off
	#	"pos": [leastNgram, highestNgram]
	config = { "lm": 1, "pos": [2, 3] }

	# setup parser and parse args
	parser = argparse.ArgumentParser(description='Transforms the sluice example data adding or removing features')
	parser.add_argument('dataref', metavar='dataref', type=str, help='Reference to the example file')
	parser.add_argument('--save', metavar='save', type=str, help='Destination for the file save')
	args = parser.parse_args()

	# load data 
	examples = loadData(args.dataref)

	# add language model and load its features
	lmModel = kenlm.Model('models/standard.arpa')
	# examples = { "Treebanks/NYT-Parsed/nyt_eng_199408.tgrep2_4831_114": examples[key] for key in ["Treebanks/NYT-Parsed/nyt_eng_199408.tgrep2_4831_114"] }

	# ---------------------------------------------------------------
	# add language model specs to examples
	# ---------------------------------------------------------------
	if config["lm"] != 0: 
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


	saveData(args.save, examples, 1);