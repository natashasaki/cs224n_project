import pickle
from transformers import BertTokenizer
from model import BertClassifier
import random
import numpy as np
import torch
from create_dataset import createDataset, preprocessForBERT, loadData, splitData
import pandas as pd
import re
import os

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def preprocessForAnalysis(data, max_len):
    input_ids = []
    attention_masks = []
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
    
    for comment in data:
    #   print(str(comment).encode('utf8'))
      encoded_sent = tokenizer.encode_plus(
          text = re.sub(r'\s+', ' ', comment).strip(),
          add_special_tokens = True,
          max_length = max_len,
          pad_to_max_length = True,
          return_attention_mask = True,
          truncation = True # truncate text that is too long
    )

    input_ids.append(encoded_sent.get('input_ids'))
    attention_masks.append(encoded_sent.get('attention_mask'))
    
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    preds_all = []
    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        preds_all = preds_all + preds.cpu().numpy().tolist()

    print(np.array(preds_all))
    
    return preds_all    


############### MAIN CODE ###############
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if os.path.exists(".\data\\analysis_dataloader.pkl"):
    train_dataloader = pickle.load(open(".\data\\analysis_dataloader.pkl", "rb"))
else:
    set_seed(123)
    data = loadData()
    X_train, y_train = splitData(data)

    # pre-process data for BERT
    MAX_LEN = 512
    train_inputs, train_masks = preprocessForBERT(X_train, max_len=MAX_LEN)
    print(len(train_inputs))
    train_labels = torch.tensor(y_train)
    train_dataloader = createDataset(train_inputs, train_masks, train_labels, batch_size=64)
    pickle.dump(train_dataloader, open(".\data\\analysis_dataloader.pkl", "wb"))

print("created dataset", flush=True)


model = BertClassifier(outputDim=6)
model.load_state_dict(torch.load(".\saved_models\exp4_stage2.model", map_location=torch.device('cpu')))
model.to(device)
print(evaluate(model, train_dataloader))

# # Get the predictions
# preds = torch.argmax(logits, dim=1).flatten()
# print(preds.cpu().numpy().tolist())

