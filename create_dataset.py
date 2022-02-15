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
        data = pd.read_csv("dataset.csv", names=header_list,)
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['condition_label'], test_size = 0.2, random_state = 123)


def text_preprocessing(text):

    # remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # TODO: add more pre-processing steps?

    return text


def preprocessForBERT(data, max_len, tokenizer):
  # Initialise empty arrays
  input_ids = []
  attention_masks = []

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

def createDataset(): 
    header_list = ["text", "condition_label", "emotion_label"]
    data = pd.read_csv("./dataset/dataset.csv", names=header_list)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
    MAX_LEN = 512

    # train-val-test split: 80-10-10
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['condition_label'], test_size = 0.2, random_state = 123)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 123)

    y_train= np.array(y_train.apply(lambda x: np.array(literal_eval(x)), 0).values.tolist())
    y_val = np.array(y_val.apply(lambda x: np.array(literal_eval(x)), 0).values.tolist())

    train_inputs, train_masks = preprocessForBERT(X_train, MAX_LEN, tokenizer)
    val_inputs, val_masks = preprocessForBERT(X_val, MAX_LEN, tokenizer)

    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    batch_size = 32

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = batch_size)
    return train_dataloader, val_dataloader

