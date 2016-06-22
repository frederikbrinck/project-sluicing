#------------------LMM-----------------#
# execute the language model antecedent
# evaluation currently using the KenML
# python library. Note that you must add
# more examples or modify the kfold factor
# in f_pos.py for this to run properly
python f_language.py data/examples-test.jsons

#------------------POS-----------------#
# note that in order to run this file
# you must add more examples or modify the 
# kfold factor in f_pos.py
python f_pos.py data/examples-test.jsons models/table

#---------------SLUICING---------------#
# create table of probabilities
python sluicing.py data/examples-test.jsons --save models/table

# evaluate score for the given model 
# using random coefficients
python sluicing.py data/examples-test.jsons --model models/table

# use cross validation for evaluating the model
# created from examples-test.jsons by making
# 10 groups of data
python sluicing.py data/examples-test.jsons --kfold 10