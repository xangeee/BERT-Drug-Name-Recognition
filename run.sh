#! /bin/bash

BASEDIR=.

# convert datasets to feature vectors
echo "Extracting features..."
python extract-features.py $BASEDIR/data/train/ train 
python extract-features.py $BASEDIR/data/devel/ test 

# # train and validate the model
python main.py
