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
    # bert classifier #BertForSequenceClassification.from_pretrained num_labels=5,
                                                    #   output_attentions=False,
                                                    #   output_hidden_states=False)

    bert_classifier = BertClassifier(outputDim=14) # 6 conditions + 8 emotions
                                                     
    bert_classifier.to(device)

    # optimiser (note, only classifier/finetuning weights will be modified)
    optimizer = AdamW(bert_classifier.parameters(), lr = learning_rate, eps=1e-8)
    epochs = 2

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
    y_integers = np.argmax(train_labels[:,:6], axis=1)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers.cpu().detach().numpy())

    class_weights = np.concatenate((class_weights, [1,1,1,1,1,1,1,1]))
    weights= torch.tensor(class_weights,dtype=torch.float)
    print(weights)      
    # push to GPU
    weights = weights.to(device)
    loss_fn = nn.BCELoss(weight=weights)# push to GPU
    # weights = weights.to(device)
    #loss_fn = nn.BCELoss()
    # loss_fn = nn.CrossEntropyLoss(weight=weights)

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
            # labels=b_labels.argmax(dim=1)
            # labels = labels.reshape((labels.shape[0]))

            # labels = b_labels.reshape((b_labels.shape[0]))
            # print(labels.shape)
            y_actual.append(b_labels)
            
            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)
            # print(logits.shape)
            # print(logits)
            y_preds.append(logits)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels.type(torch.float))
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

        torch.save(model.state_dict(), f'./exp3_online/exp3_weights_bs_64_lr_5e-4_epoch_{epoch_i}.model')
        print("\n")

    print("Training complete!")
    #print(y_actual)
    #print(y_preds)
    return y_actual, y_preds



def evaluate(model, val_dataloader, mode=None, epoch="None"):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    loss_fn = nn.BCELoss()

    labels_all = []
    preds_all = []
    condition_labels = []
    pred_conds = []
    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
 #       print(b_labels)
        # labels=b_labels.argmax(dim=1)
        # labels = labels.reshape((labels.shape[0]))

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids.to(device), b_attn_mask.to(device))

        labels = b_labels
        # Compute loss
        loss = loss_fn(logits, labels.type(torch.float))
        val_loss.append(loss.item())

        # Get the predictions
        # preds = torch.argmax(logits, dim=1).flatten()
        threshold = 0.5
        one_hot = logits > threshold
        preds = torch.from_numpy(one_hot.cpu().numpy()).to(device)
        preds.to(device)
        labels.to(device)
        labels.cuda()
        preds.cuda()
#        print(preds)
#        print(labels.get_device())
#        print(preds.get_device())
        #print((preds == labels))
        # Calculate the accuracy rate
        #accuracy = (preds == labels).numpy().mean() * 100
        #print(preds)
        #print(labels)
        accuracy = (preds == labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)
#        print(preds.cpu().numpy().tolist())
        preds_all = preds_all + preds.cpu().numpy().tolist()
        labels_all = labels_all + labels.cpu().numpy().tolist()
        conditions = labels[:,:6].argmax(dim=1)
        condition_labels = condition_labels + conditions.cpu().numpy().tolist()
        pred_conditions = preds.cpu().numpy()[:,:6].argmax(axis=1)
        pred_conds = pred_conds + pred_conditions.tolist()
    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)    

    labels = [0, 1, 2, 3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    target_names = ["depression", "anxiety", "bipolar", "addiction", "adhd", "none","anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust" ]
    print(classification_report(np.array(labels_all), np.array(preds_all), labels=labels, target_names = target_names))
    
    #condition_labels = labels_all[:,0:6]
    #pred_cond = preds_all[:, :6]
    if mode:
        cM = confusion_matrix(condition_labels, pred_conds)
                                    
    #displayClasses = [i for i in range(8)]
        condition_names = ["depression", "anxiety", "bipolar", "addiction", "adhd", "none"]
        disp = ConfusionMatrixDisplay(confusion_matrix=cM, display_labels=condition_names)
        disp.plot()
        disp.figure_.savefig(f"confusion_exp3_custom_{epoch}_{mode}.png" , dpi=300)

    return val_loss, val_accuracy    


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
        b_input_ids.to(device)
        b_attn_mask.to(device)
        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids.to(device), b_attn_mask.to(device))
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs



############### MAIN CODE ###############
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if os.path.exists("./data/train_dataloader.pkl"):
    train_dataloader = pickle.load(open("./data/train_dataloader.pkl", "rb"))
    val_dataloader = pickle.load(open("./data/val_dataloader.pkl", "rb"))
    test_dataloader = pickle.load(open("./data/test_dataloader.pkl", "rb"))

    train_labels = pickle.load(open("./data/train_labels.pkl", "rb"))
    val_labels = pickle.load(open("./data/val_labels.pkl", "rb"))
    test_labels = pickle.load(open("./data/test_labels.pkl", "rb"))
else:
    set_seed(123)    # Set seed for reproducibility
    data = loadData()
    X_train, y_train, X_val, y_val, X_test, y_test = splitData(data)

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
    
    pickle.dump(train_labels, open("./data/train_labels_exp3.pkl", "wb"))
    pickle.dump(val_labels, open("./data/val_labels_exp3.pkl", "wb"))
    pickle.dump(test_labels, open("./data/test_labels_exp3.pkl", "wb"))
    pickle.dump(train_dataloader, open("./data/train_dataloader_exp3.pkl", "wb"))
    pickle.dump(val_dataloader, open("./data/val_dataloader_exp3.pkl", "wb"))
    pickle.dump(test_dataloader, open("./data/test_dataloader_exp3.pkl", "wb"))
print("created dataset")

learning_rate = 5e-4
#bert_classifier, optimizer, scheduler = initialize(learning_rate)
print("initialized model")
 
# train and evaluate model 
#y_actual, y_preds = train(bert_classifier, optimizer, train_labels, scheduler, train_dataloader, val_dataloader, epochs=2, evaluation=True)

# load model if saved prioir
model = BertClassifier(outputDim=14)
model.load_state_dict(torch.load("./exp3_epoch_1")) #, map_location=torch.device('cpu')))
model.to(device)

#print("epoch 0 model")
#print(evaluate(bert_classifier, val_dataloader))

#print(evaluate(model, val_dataloader))
#print("done with val set metrics")

print("calculating test set metrics")
print(evaluate(model, test_dataloader, "test", "1"))
print("done")
#model = BertClassifier(outputDim=14)
#model.load_state_dict(torch.load("./local_exp3/exp3_epoch_1")) #, map_location=torch.device('cpu')))
#model.to(device)

print("calculating val set metrics")
#print("epoch 1 model")
print(evaluate(model, val_dataloader, "val", "1"))
print("done with val set metrics")

#model = BertClassifier(outputDim=14)
#model.load_state_dict(torch.load("./exp3_weights_bs_64_lr_5e-4_epoch_0.model")) #, map_location=torch.device('cpu')))
#model.to(device)

#print("calculating val set metrics")
#print("epoch 1 model")
#print(evaluate(model, val_dataloader, "val", "0"))
#print("done with val set metrics")

#print("calculating test set metrics")
#print(evaluate(model, test_dataloader, "test", "0"))
#print("done")

#test set metrics (only do this a few times max)
#print("calculating test set metrics")
#print(evaluate(model, test_dataloader))
#print("done with test set metrics")
