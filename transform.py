import kenlm
import json
from lib.data import loadData, saveData
from features.f_language import extractFeatures

if __name__ == '__main__':
	import argparse

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

	saveData(args.save, examples, 1);