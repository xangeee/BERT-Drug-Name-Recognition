#!/usr/bin/env python3
import sys
from contextlib import redirect_stdout
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from torch import cuda
import dataset
import validate
from seqeval.metrics import classification_report
import os

MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
     
def instances(fi):
    xseq = []
    yseq = []
    
    for line in fi:
        line = line.strip('\n')
        if not line:
            # An empty line means the end of a sentence.
            # Return accumulated sequences, and reinitialize.
            yield xseq, yseq
            xseq = []
            yseq = []
            continue

        # Split the line with TAB characters.
        fields = line.split('\t')
        
        # Append the item features to the item sequence.
        # fields are:  0=sid, 1=form, 2=span_start, 3=span_end, 4=tag, 5...N = features
        item = fields[5:]        
        xseq.append(item)
        
        # Append the label to the label sequence.
        yseq.append(fields[4])


def predict(tokenizer,ids_to_labels):

    sentence = "Sildenafil may markedly increase the hypotensive effects of isosorbide mononitrate"

    inputs = tokenizer(sentence.split(),
                        is_pretokenized=True, 
                        return_offsets_mapping=True, 
                        padding='max_length', 
                        truncation=True, 
                        max_length=MAX_LEN,
                        return_tensors="pt")

    # move to gpu
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model(ids, attention_mask=mask)
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

    prediction = []
    for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
    #only predictions on first word pieces are important
        if mapping[0] == 0 and mapping[1] != 0:
            prediction.append(token_pred[1])
        else:
            continue

    print(sentence.split())
    print(prediction)
    
def readData(path):
    data = pd.read_csv(path, encoding='unicode_escape')
    print(data.head())
    
    print("Number of tags: {}".format(len(data.BIOtag.unique())))

    labels_to_ids = {k: v for v, k in enumerate(data.BIOtag.unique())}
    ids_to_labels = {v: k for v, k in enumerate(data.BIOtag.unique())}
    print(labels_to_ids)
    
    # let's create a new column called "sentence" which groups the words by sentence 
    data['sentence'] = data[['sentenceId','word','BIOtag']].groupby(['sentenceId'])['word'].transform(lambda x: ' '.join(x))
    # let's also create a new column called "word_labels" which groups the tags by sentence 
    data['word_labels'] = data[['sentenceId','word','BIOtag']].groupby(['sentenceId'])['BIOtag'].transform(lambda x: ','.join(x))
    
    data = data[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
    print("converted sentence\n")
    print(data.iloc[0].sentence)
    print(data.iloc[0].word_labels)
    return data,labels_to_ids,ids_to_labels

def prepareDatasets(tokenizer):
    
    train_dataset,train_labels_to_ids,train_ids_to_labels=readData('drug_data_train.csv')
    test_dataset,test_labels_to_ids,test_ids_to_labels=readData('drug_data_test.csv')
    print("Training set: {}".format(train_dataset.shape))
    print("Testing set: {}".format(test_dataset.shape))  
   
    training_set = dataset.dataset(train_dataset, tokenizer, MAX_LEN,train_labels_to_ids)
    testing_set = dataset.dataset(test_dataset, tokenizer, MAX_LEN,test_labels_to_ids)
    return training_set,testing_set,train_labels_to_ids,train_ids_to_labels

def sanity_check(training_set):
    
    inputs = training_set[2]
    input_ids = inputs["input_ids"].unsqueeze(0)
    attention_mask = inputs["attention_mask"].unsqueeze(0)
    labels = inputs["labels"].unsqueeze(0)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    print(input_ids)
    print(attention_mask)
    print(labels)
    
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    initial_loss = outputs[0]
    print("initial loss:",initial_loss)
    
    tr_logits = outputs[1]
    print("logits of the NN:",tr_logits.shape)
    
    
    
# Defining the training function on the 80% of the dataset for tuning the bert model
def train(training_loader,model,device,optimizer):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
   
    for idx, batch in enumerate(training_loader):
       
        ids = batch['input_ids'].to(device, dtype = torch.int32)
        mask = batch['attention_mask'].to(device, dtype = torch.int32)
        labels = batch['labels'].to(device, dtype = torch.int32)
        
        loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        
        if idx % 100==0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss per 100 training steps: {loss_step}")
           
        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")
 
def save(tokenizer,model):
    #save the model
    directory = "./model"

    if not os.path.exists(directory):
        os.makedirs(directory)

    # save vocabulary of the tokenizer
    tokenizer.save_vocabulary(directory)
    # save the model weights and its configuration file
    model.save_pretrained(directory)
    print('Model saved')  
    
     
if __name__ == '__main__':

    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    training_set,testing_set,labels_to_ids,ids_to_labels=prepareDatasets(tokenizer)
   
    # print("training_set:",training_set[0])
   
    #define the corresponding PyTorch dataloaders
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    
    #defining the model
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(labels_to_ids))
    print(model.to(device))
    
    sanity_check(training_set)
    
    #define the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    
    #train the model
    for epoch in range(EPOCHS):
        print("Training epoch:",epoch + 1)
        train(training_loader,model,device,optimizer)
    
    #validate the model 
    labels, predictions = validate.valid(model, testing_loader)
    print(classification_report(labels, predictions))
    
    #predict with example
    predict(tokenizer,ids_to_labels)
    save(tokenizer,model)
    
   