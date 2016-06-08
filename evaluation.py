from nltk.tree import *
import nltk

prev = 0
def filt(x):
	return x.label()=='S'

commaSeparated = False
def splitTree(t, i, sentences):	
	global commaSeparated

	if t.height() < 3:
		print("tree:", t, t.height(), t.label())
		if not sentences[i]:
			sentences[i] = []
		sentences[i].append(t)

		return


	for subtree in t:
		if type(subtree) == nltk.tree.ParentedTree:
			if subtree.left_sibling() != None and subtree.left_sibling().label() == ",":
				i += 1
				commaSeparated = True
			if subtree.label() == "NP" and subtree.right_sibling() != None and subtree.right_sibling().label() == "VP" and commaSeparated == False:
				i += 1
			elif subtree.label() == "VP" and subtree.left_sibling() != None and subtree.left_sibling().label() == "NP":
				i += 1
				commaSeparated = False

			splitTree(subtree, i, sentences)

	return sentences

fp = open("exam.out","r")
for line in fp:
	t = ParentedTree.fromstring(line);
	print "----"
	print "\n\n"
	print "----"
	print t

	dict = splitTree(t, 0, [])
	print dict


	#splitTree(t)

		# tags = []
		# words = []
		# for pos in subtree.pos():
		# 	tuple = list(pos)
		# 	words.append(tuple[0])
		# 	tags.append(tuple[1])

		# print " ".join(words)
		# print " ".join(tags)