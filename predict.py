#!/usr/bin/env python3

import sys
from ML_model import *

def instances(fi):
    xseq = []
    toks = []
    
    for line in fi:
        line = line.strip('\n')
        if not line:
            # An empty line means the end of a sentence.
            # Return accumulated sequences, and reinitialize.
            yield xseq, toks
            xseq = []
            toks = []
            continue

        # Split the line with TAB characters.
        fields = line.split('\t')
        # Append the item features to the item sequence.
        # fields are:  0=sid, 1=form, 2=span_start, 3=span_end, 4=tag, 5...N = features
        item = fields[5:]        
        xseq.append(item)

        # Append token information (needed to produce the appropriate output)
        toks.append([fields[0],fields[1],fields[2],fields[3]])


