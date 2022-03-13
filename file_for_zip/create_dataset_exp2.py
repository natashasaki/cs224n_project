import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import re
import numpy as np
from ast import literal_eval

def text_preprocessing(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocessForBERT(data, max_len):
  input_ids = []
  attention_masks = []
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)

  for comment in data:
    encoded_sent = tokenizer.encode_plus(
        text = text_preprocessing(comment),
        add_special_tokens = True,
        max_length = max_len,
        pad_to_max_length = True,
        return_attention_mask = True,
        truncation = True
    )

    input_ids.append(encoded_sent.get('input_ids'))
    attention_masks.append(encoded_sent.get('attention_mask'))

  input_ids = torch.tensor(input_ids)
  attention_masks = torch.tensor(attention_masks)

  return input_ids, attention_masks


def loadData():
    header_list = ["input", "condition_label"]
    data = pd.read_csv("./dataset/dataset_exp2.csv", on_bad_lines='skip', names=header_list)
    data["condition_label"] = data["condition_label"].apply(lambda x: literal_eval(x))

    return data


def splitData(data):
    X_train, X_test, y_train, y_test = train_test_split(data['input'], data['condition_label'], test_size = 0.2, random_state = 123)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 123)
    y_train= np.array(y_train.apply(lambda x: np.array(x), 0).values.tolist())
    y_val = np.array(y_val.apply(lambda x: np.array(x), 0).values.tolist())
    y_test= np.array(y_test.apply(lambda x: np.array(x), 0).values.tolist())
 
    return X_train, y_train, X_val, y_val, X_test, y_test

def createDataset(inputs, masks, labels, batch_size=32):
    data = TensorDataset(inputs, masks, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler = sampler, batch_size = batch_size)
    return dataloader
