from transformers import AdamW, get_linear_schedule_with_warmup
from model import BertClassifier
import random
import time
import numpy as np
import torch.nn as nn
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import BertForSequenceClassification
from create_dataset import createDataset, preprocessForBERT, loadData, splitData
from matplotlib import pyplot as plt

def initialize():
    # bert classifier #BertForSequenceClassification.from_pretrained num_labels=5,
                                                    #   output_attentions=False,
                                                    #   output_hidden_states=False)
    bert_classifier = BertClassifier(outputDim=6)
                                                     
    #BertClassifier(outputDim=5)
    bert_classifier.to(device)

    # optimiser (note, only classifier/finetuning weights will be modified)
    optimizer = AdamW(bert_classifier.parameters(), lr = 5e-5, eps=1e-8)

    epochs = 3 #TODO: consider changing -- recommended # epochs for BERT between 2 and 4 (Sun et al., 2020)

    # scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * epochs)
    
    return bert_classifier, optimizer, scheduler



def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)



def train(model, optimizer,train_labels, scheduler, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    print(np.unique(train_labels.values))
    print(np.array(train_labels.values))
    
    labs = train_labels.cpu().detach().numpy()
    unique = np.unique(labs ,axis=0)
    # print(np.array(train_labels).shape)
    y_integers = np.argmax(train_labels, axis=1)
    print(y_integers.shape)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers.cpu().detach().numpy())

    # class_weights = compute_class_weight(class_weight ='balanced', classes=unique, y=np.array(train_labels.values))
    print(class_weights)
    weights= torch.tensor(class_weights,dtype=torch.float)

    # push to GPU
    weights = weights.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)

    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()
        y_actual = []
        y_preds = []

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            labels=b_labels.argmax(dim=1)
            labels = labels.reshape((labels.shape[0]))
            y_actual.append(labels)
            
            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)
            y_preds.append(logits)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            print("val set: ")
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)

        torch.save(model.state_dict(), './saved_models/most_recent_baseline.model')
        print("\n")

    print("Training complete!")
    print(y_actual)
    print(y_preds)
    return y_actual, y_preds



def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    loss_fn = nn.CrossEntropyLoss()
    labels_all = []
    preds_all = []
    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        labels=b_labels.argmax(dim=1)
        labels = labels.reshape((labels.shape[0]))

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)
        print(preds.cpu().numpy().tolist())
        preds_all = preds_all + preds.cpu().numpy().tolist()
        labels_all = labels_all + labels.cpu().numpy().tolist()

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)    
    print(labels_all)
    print(preds_all)
    print(classification_report(np.array(labels_all), np.array(preds_all), labels=[0, 1, 2, 3,4,5], target_names = ["depression", "anxiety", "bipolar", "addiction", "adhd", "none"]))
    cM = confusion_matrix(labels_all, preds_all)

    displayClasses = [i for i in range(6)]

    disp = ConfusionMatrixDisplay(confusion_matrix=cM, display_labels=displayClasses)
    disp.plot()
    disp.figure_.savefig('confusion_most_recent_baseline.png', dpi=300)

    return val_loss, val_accuracy    

from sklearn.metrics import accuracy_score, roc_curve, auc

def evaluate_roc(probs, y_true):
    preds = probs.argmax(axis=1) #probs[:, 1]
    print(preds)
    print(y_true)
    fpr, tpr, threshold = roc_curve(y_true.detach().numpy(), preds.detach().numpy())
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
       
    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    
    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

import torch.nn.functional as F


def make_predictions(model, dataloader):
    """Uses model passed in to predict probabilities on the val/test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs





############### MAIN CODE ###############
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

set_seed(123)    # Set seed for reproducibility
data = loadData()
X_train, y_train, X_val, y_val, X_test, y_test = splitData(data)
# train_dataloader, val_dataloader, test_dataloader = createDataset()

# pre-process data for BERT
MAX_LEN = 512
train_inputs, train_masks = preprocessForBERT(X_train, max_len=MAX_LEN)
val_inputs, val_masks = preprocessForBERT(X_val, max_len=MAX_LEN)
test_inputs, test_masks = preprocessForBERT(X_test, max_len = MAX_LEN)

train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)
test_labels = torch.tensor(y_test)
train_dataloader = createDataset(train_inputs, train_masks, train_labels, batch_size=32)
val_dataloader = createDataset(val_inputs, val_masks, val_labels, batch_size=32)
test_dataloader = createDataset(test_inputs, test_masks, test_labels, batch_size=32)
print("created dataset")
#bert_classifier, optimizer, scheduler = initialize()
#print("initialized model")
 
# train and evaluate model 
#y_actual, y_preds = train(bert_classifier, optimizer, train_labels, scheduler, train_dataloader, val_dataloader, epochs=2, evaluation=True)

# predictt probabilities on val set
#probs = make_predictions(bert_classifier, val_dataloader)
#print(probs.shape)

# load model
model = BertClassifier(outputDim=6)
model.load_state_dict(torch.load("./saved_models/most_recent_baseline.model", map_location=torch.device('cpu')))
model.to(device)
print(evaluate(model, val_dataloader))


# print("calculating train set metrics")
# # train set metrics
# probs = make_predictions(bert_classifier, train_dataloader)
# probs = probs.argmax(axis=1)
# train_labels = train_labels.argmax(axis=1)
# f1_rec_prec_train = classification_report(train_labels, probs, labels=[0, 1, 2, 3,4,5], target_names = ["depression", "anxiety", "bipolar", "addiction", "adhd", "none"])
# accuracy_train = accuracy_score(train_labels, probs)
# print(f1_rec_prec_train)
# print(accuracy_train)

# print("calculating val set metrics")
# # val set metrics 
# probs = make_predictions(bert_classifier, val_dataloader)
# probs = probs.argmax(axis=1)
# val_labels = val_labels.argmax(axis=1)
# f1_rec_prec_val = classification_report(val_labels, probs, labels=[0, 1, 2, 3,4,5], target_names = ["depression", "anxiety", "bipolar", "addiction", "adhd", "none"])
# accuracy_val= accuracy_score(val_labels, probs)
# print(f1_rec_prec_val)
# print(accuracy_val)


# # test set metrics (only do this a few times max)
# print("calculating test set metrics")
# probs = make_predictions(bert_classifier, test_dataloader)
# probs = probs.argmax(axis=1)
# test_labels = test_labels.argmax(axis=1)
# f1_rec_prec_test = classification_report(test_labels, probs, labels=[0, 1, 2, 3,4,5], target_names = ["depression", "anxiety", "bipolar", "addiction", "adhd", "none"])
# accuracy_test = accuracy_score(test_labels, probs)
# print(f1_rec_prec_test)
# print(accuracy_test) 

# print("done on test set")
