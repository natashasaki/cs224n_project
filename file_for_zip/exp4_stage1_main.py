from transformers import AdamW, get_linear_schedule_with_warmup
from model_exp3 import BertClassifier
import random
import time
import pickle
import numpy as np
import torch.nn as nn
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
from create_dataset import createDataset, preprocessForBERT, loadData, splitData
from matplotlib import pyplot as plt
import os

def initialize(learning_rate=5e-5):
    bert_classifier = BertClassifier(outputDim=8)
    bert_classifier.to(device)

    optimizer = AdamW(bert_classifier.parameters(), lr = learning_rate, eps=1e-8)
    epochs = 2
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * epochs)
    
    return bert_classifier, optimizer, scheduler



def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, optimizer, train_labels, scheduler, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    print("Start training...\n")

    class_weights = [1, 1, 1, 1, 1, 1, 1, 1]
    weights= torch.tensor(class_weights,dtype=torch.float)
    print(weights)
    weights = weights.to(device)
    loss_fn = nn.BCELoss(weight=weights)

    for epoch_i in range(epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        t0_epoch, t0_batch = time.time(), time.time()

        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()
        y_actual = []
        y_preds = []

        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            y_actual.append(b_labels)

            model.zero_grad()

            logits = model(b_input_ids, b_attn_mask)
            y_preds.append(logits)

            loss = loss_fn(logits, b_labels.type(torch.float))
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if (step % 50 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch

                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        if evaluation == True:
            print("val set: ")
            val_loss, val_accuracy = evaluate(model, val_dataloader)
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)

        torch.save(model.state_dict(), './saved_models/exp4_stage1.model')
        print("\n")

    print("Training complete!")
    return y_actual, y_preds



def evaluate(model, val_dataloader, mode=None, epoch="None"):
    model.eval()
    val_accuracy = []
    val_loss = []
    loss_fn = nn.BCELoss()

    labels_all = []
    preds_all = []
    for batch in val_dataloader:
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(b_input_ids.to(device), b_attn_mask.to(device))

        labels = b_labels
        loss = loss_fn(logits, labels.type(torch.float))
        val_loss.append(loss.item())

        threshold = 0.5
        one_hot = logits > threshold
        preds = torch.from_numpy(one_hot.cpu().numpy()).to(device)
        preds.to(device)
        labels.to(device)
        labels.cuda()
        preds.cuda()
        
        accuracy = (preds == labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)
        
        preds_all = preds_all + preds.cpu().numpy().tolist()
        labels_all = labels_all + labels.cpu().numpy().tolist()
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)    

    labels = [0, 1, 2, 3, 4, 5, 6, 7]
    target_names = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust" ]
    print(classification_report(np.array(labels_all), np.array(preds_all), labels=labels, target_names = target_names))
    return val_loss, val_accuracy    


############### MAIN CODE ###############
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if os.path.exists("./data/train_dataloader_exp4_1_64.pkl"):
    train_dataloader = pickle.load(open("./data/train_dataloader_exp4_1_64.pkl", "rb"))
    val_dataloader = pickle.load(open("./data/val_dataloader_exp4_1_64.pkl", "rb"))
    test_dataloader = pickle.load(open("./data/test_dataloader_exp4_1_64.pkl", "rb"))

    train_labels = pickle.load(open("./data/train_labels_exp4_1_64.pkl", "rb"))
    val_labels = pickle.load(open("./data/val_labels_exp4_1_64.pkl", "rb"))
    test_labels = pickle.load(open("./data/test_labels_exp4_1_64.pkl", "rb"))
else:
    set_seed(123)
    data = loadData()
    X_train, y_train, X_val, y_val, X_test, y_test = splitData(data, False)

    # pre-process data for BERT
    MAX_LEN = 512
    train_inputs, train_masks = preprocessForBERT(X_train, max_len=MAX_LEN)
    val_inputs, val_masks = preprocessForBERT(X_val, max_len=MAX_LEN)
    test_inputs, test_masks = preprocessForBERT(X_test, max_len = MAX_LEN)

    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)
    test_labels = torch.tensor(y_test)
        
    train_dataloader = createDataset(train_inputs, train_masks, train_labels, batch_size=64)
    val_dataloader = createDataset(val_inputs, val_masks, val_labels, batch_size=64)
    test_dataloader = createDataset(test_inputs, test_masks, test_labels, batch_size=64)
    
    pickle.dump(train_labels, open("./data/train_labels_exp4_1_64.pkl", "wb"))
    pickle.dump(val_labels, open("./data/val_labels_exp4_1_64.pkl", "wb"))
    pickle.dump(test_labels, open("./data/test_labels_exp4_1_64.pkl", "wb"))
    pickle.dump(train_dataloader, open("./data/train_dataloader_exp4_1_64.pkl", "wb"))
    pickle.dump(val_dataloader, open("./data/val_dataloader_exp4_1_64.pkl", "wb"))
    pickle.dump(test_dataloader, open("./data/test_dataloader_exp4_1_64.pkl", "wb"))
print("created dataset")

learning_rate = 5e-4
bert_classifier, optimizer, scheduler = initialize(learning_rate)
print("initialized model")
 
y_actual, y_preds = train(bert_classifier, optimizer, train_labels, scheduler, train_dataloader, val_dataloader, epochs=1, evaluation=True)

model = BertClassifier(outputDim=8)
model.load_state_dict(torch.load("./saved_models/exp4_stage1.model"))
model.to(device)

print("calculating test set metrics")
print(evaluate(model, test_dataloader, "test", "1"))
print("done")

print("calculating val set metrics")
print(evaluate(model, val_dataloader, "val", "1"))
print("done with val set metrics")
