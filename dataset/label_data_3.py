import pandas as pd
import numpy as np
from collections import defaultdict
import csv


# gets labels (one hot vec) for emotions and condition
def combine_labels(reddit_text):

    # text_subreddit_file = open(reddit_text)
    w = csv.writer(open("dataset_labelled_exp_3.csv", "a"))

    with open(reddit_text, newline="", encoding='Latin1') as csvfile:

        # TODO: might need to change this depending on structure of csv
        reader = csv.reader(csvfile) 
        for row in reader:
            try:
                text = row[0] 
                condition_label = row[1]
                emotion_label = row[2]

                cond_emo_label = condition_label[:len(condition_label)-1] + "," + emotion_label[1:]
                        
                w.writerow([text, cond_emo_label])
            except:
                continue
    print("done processing, created dataset")
    return 


# set write_to_disk = True if want to save to file
combine_labels("./dataset_labelled.csv") # outputs to file: dataset.csv