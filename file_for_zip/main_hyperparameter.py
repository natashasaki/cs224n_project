from transformers import AdamW, get_linear_schedule_with_warmup
from model import BertClassifier
import random
import time
import numpy as np
import torch.nn as nn
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertForSequenceClassification
from create_dataset import createDataset, preprocessForBERT, loadData, splitData
from matplotlib import pyplot as plt

def initialize(learning_rate):
    bert_classifier = BertClassifier(outputDim=6) # 6 condition labels
    bert_classifier.to(device)

    optim = AdamW(bert_classifier.parameters(), lr = learning_rate, eps=1e-8)
    epochs = 3
    lr_schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * epochs)
    
    return bert_classifier, optim, lr_schedule



def set_seed(seed_value=25):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)



def train(model, optimizer, train_labels, scheduler, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    print("Started Training Model...\n")
    y_integers = np.argmax(train_labels, axis=1)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers.cpu().detach().numpy())
    weights= torch.tensor(class_weights,dtype=torch.float)

    weights = weights.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)

    for epoch_i in range(epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*60)

        t0_epoch, t0_batch = time.time(), time.time()

        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()
    
        y_preds = []

        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            labels=b_labels.argmax(dim=1)
            labels = labels.reshape((labels.shape[0]))
            
            model.zero_grad()

            logits = model(b_input_ids, b_attn_mask)
            y_preds.append(logits)

            loss = loss_fn(logits, labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch

                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*60)
        if evaluation == True:
            print("val set: ")
            val_loss, val_accuracy = evaluate(model, val_dataloader)
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*60)

        torch.save(model.state_dict(), f"./saved_models/baseline_epoch{epoch_i}.model")
        print("\n")

    print("Done training model!  Checkpoints Saved!")
    return y_preds



def evaluate(model, val_dataloader):
    model.eval()

    val_accuracy = []
    val_loss = []
    loss_fn = nn.CrossEntropyLoss()

    for batch in val_dataloader:
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        labels=b_labels.argmax(dim=1)
        labels = labels.reshape((labels.shape[0]))

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        loss = loss_fn(logits, labels)
        val_loss.append(loss.item())

        preds = torch.argmax(logits, dim=1).flatten()

        accuracy = (preds == labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)    
    
    return val_loss, val_accuracy    


import torch.nn.functional as F
def make_predictions(model, dataloader):
    model.eval() # no dropout!

    all_logits = []

    for batch in dataloader:
        ids, masks = tuple(t.to(device) for t in batch)[:2]

        with torch.no_grad():
            logits = model(ids, masks)
        all_logits.append(logits)
    
    all_logits = torch.cat(all_logits, dim=0)

    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs

############### MAIN CODE ###############
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

set_seed(25)
data = loadData()
X_train, y_train, X_val, y_val, X_test, y_test = splitData(data)

MAX_LEN = 512
train_inputs, train_masks = preprocessForBERT(X_train, max_len=MAX_LEN)
val_inputs, val_masks = preprocessForBERT(X_val, max_len=MAX_LEN)
test_inputs, test_masks = preprocessForBERT(X_test, max_len = MAX_LEN)

train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)
test_labels = torch.tensor(y_test)


learning_rates = [5e-5, 5e-3, 5e-1, .5, .95]
batch_sizes = [32, 64, 128]
training_epochs = [2, 4, 6, 8, 10]

val_accuracies = []

for lr in learning_rates:
    for bs in batch_sizes:
        for e in training_epochs:
            train_dataloader = createDataset(train_inputs, train_masks, train_labels, batch_size=bs)
            val_dataloader = createDataset(val_inputs, val_masks, val_labels, batch_size=bs)
            test_dataloader = createDataset(test_inputs, test_masks, test_labels, batch_size=bs)
            print("created dataset")
            bert_classifier, optimizer, scheduler = initialize(lr)
            print("initialized model")
            
            y_preds = train(bert_classifier, optimizer, train_labels, scheduler, train_dataloader, val_dataloader, epochs=e, evaluation=True)

            print("calculating val set metrics")
            probs = make_predictions(bert_classifier, val_dataloader)
            probs = probs.argmax(axis=1)
            val_labels = val_labels.argmax(axis=1)
            f1_rec_prec_val = classification_report(val_labels, probs,output_dict=True, labels=[0, 1, 2, 3,4,5], target_names = ["depression", "anxiety", "bipolar", "addiction", "adhd", "none"])
            accuracy_val= accuracy_score(val_labels, probs)
            f = open("./metrics/dict_val.txt","w")

            f.write( str(f1_rec_prec_val))

            f.close()

            print(f1_rec_prec_val)
            print(accuracy_val)

            val_accuracies.append((accuracy_val, lr, bs, e))

            print("calculating test set metrics")
            probs = make_predictions(bert_classifier, test_dataloader)
            probs = probs.argmax(axis=1)
            test_labels = test_labels.argmax(axis=1)
            f1_rec_prec_test = classification_report(test_labels, probs,output_dict=True, labels=[0, 1, 2, 3,4,5], target_names = ["depression", "anxiety", "bipolar", "addiction", "adhd", "none"])
            accuracy_test = accuracy_score(test_labels, probs)
            print(f1_rec_prec_test)
            print(accuracy_test) 
            f = open("./metrics/dict_test.txt","w")
            f.write( str(f1_rec_prec_test) )

            f.close()
            print("done on test set")

sorted(val_accuracies, key=lambda x: x[0])
print(val_accuracies[0])
