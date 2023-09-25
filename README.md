# BERT-Drug-Name-Recognition
This repository is dedicated to addressing Drug Name Entity Recognition using BERT.
The code is based on the following notebook: [Custom Named Entity Recognition with BERT](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=CFRDM8WsQXvL),
which is available on the GitHub repository maintained by Niels Rogge at [here](https://github.com/NielsRogge/Transformers-Tutorials).

The implemented solution to tackle this challenge uses a machine learning classification approach. 
Each token is classified into one of the following classes: {'O': 0, 'B-drug': 1, 'B-drug_n': 2, 'B-group': 3, 'I-group': 4, 'I-drug_n': 5, 'B-brand': 6, 'I-brand': 7, 'I-drug': 8}
