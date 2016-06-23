#------------------LMM-----------------#
# execute the language model antecedent
# evaluation currently using the KenML
# python library. Note that you must add
# more examples or modify the kfold factor
# in f_pos.py for this to run properly
python features/f_language.py data/examples-test.jsons

#------------------POS-----------------#
# note that in order to run this file
# you must add more examples or modify the 
# kfold factor in f_pos.py
python features/f_pos.py data/examples-test.jsons --model models/table

#---------------SLUICING---------------#
# run all features enabled in sluicing
python sluicing.py data/examples-test.jsons 