# create table of probabilities
python sluicing.py data/examples-test.jsons --save data/table

# evaluate score for the given model 
# using random coefficients
python sluicing.py data/examples-test.jsons --model data/table

# use cross validation for evaluating the model
# created from examples-test.jsons by making
# 10 groups of data
python sluicing.py data/examples-test.jsons --kfold 10