import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import re
import numpy as np
from ast import literal_eval

def RedditToConditionDataset(Dataset): 
    def __init__(self, path, max_len, ):
        header_list = ["text", "condition_label", "emotion_label"]
        data = pd.read_csv("./dataset/dataset_partial.csv", names=header_list,)
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['condition_label'], test_size = 0.2, random_state = 123)


def text_preprocessing(text):

    # remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # TODO: add more pre-processing steps?

    return text


def preprocessForBERT(data, max_len):
  # Initialise empty arrays
  input_ids = []
  attention_masks = []
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)

  # Encode_plus with above processing
  for comment in data:
    encoded_sent = tokenizer.encode_plus(
        text = text_preprocessing(comment),
        add_special_tokens = True,
        max_length = max_len,
        pad_to_max_length = True,
        return_attention_mask = True,
        truncation = True # truncate text that is too long
    )

    input_ids.append(encoded_sent.get('input_ids'))
    attention_masks.append(encoded_sent.get('attention_mask'))
  
  # Convert list to tensors
  input_ids = torch.tensor(input_ids)
  attention_masks = torch.tensor(attention_masks)

  return input_ids, attention_masks


def loadData(): 
    """
      Reads in data from csv file
    """
    header_list = ["text", "condition_label", "emotion_label"]
    data = pd.read_csv("./dataset/dataset_partial.csv", on_bad_lines='skip', names=header_list)
        # data = pd.read_csv("./dataset/dataset.csv", on_bad_lines='skip', names=header_list)

    print(data.head())
    dist = data.groupby('condition_label')['text'].nunique()
    print(dist)
    N = 100
    dep= data.loc[data['condition_label'] == "[1,0,0,0,0,0]"].head(100)
    print(dep)
    anx= data.loc[data['condition_label'] == "[0,1,0,0,0, 0]"].head(200)
    print(anx)
    bp= data.loc[data['condition_label'] == "[0,0,1,0,0, 0]"].head(100)
    print(bp)
    adhd= data.loc[data['condition_label'] == "[0,0,0,0,1,0]"].head(200)
    print(adhd)
    add= data.loc[data['condition_label'] == "[0,0,0,1,0,0]"].head(100)
    print(add)
    oth= data.loc[data['condition_label'] == "[0,0,0,0,0,1]"].head(200)
    print(oth)

    frames = [dep, anx, bp, adhd, add, oth]

    result = pd.concat(frames)
    print(result)
   # return data
    return data


def splitData(data):
    """
      Splits dataset into train, dev, test set
    """

    # train-val-test split: 80-10-10
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['condition_label'], test_size = 0.2, random_state = 123)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 123)
    y_train= np.array(y_train.apply(lambda x: np.array(literal_eval(x)), 0).values.tolist())
    y_val = np.array(y_val.apply(lambda x: np.array(literal_eval(x)), 0).values.tolist())
    y_test= np.array(y_test.apply(lambda x: np.array(literal_eval(x)), 0).values.tolist())
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def createDataset(inputs, masks, labels, batch_size=32):
    data = TensorDataset(inputs, masks, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler = sampler, batch_size = batch_size)
    return dataloader

    # val_data = TensorDataset(val_inputs, val_masks, val_labels)
    # val_sampler = RandomSampler(val_data)
    # val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = batch_size)
    
    # test_data = TensorDataset(test_inputs, test_masks, test_labels)
    # test_sampler = RandomSampler(test_data)
    # test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size = batch_size)
        
    # return train_dataloader, val_dataloader, test_dataloader

