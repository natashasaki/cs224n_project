import pandas as pd
import numpy as np
from collections import defaultdict
import csv


# for emotion: we care abt 1-5
def getEmotionDict(lexicon_path, write_to_disk = False):

    emo_dict = defaultdict(list)
    emotion_lexicon = open(lexicon_path)
    
    # only for writing to file
    if write_to_disk:
        w = csv.writer(open("emo_dict.csv", "w"))

    for i, line in enumerate(emotion_lexicon): # 10 sentiments per word
        ind = i % 10
        word, emotion, value = line.split() #Split line into a tuple.

        if emotion not in ['positive', 'negative']:
            emo_dict[word].append(value)  #Add tuple values to dictionary.
        
        if write_to_disk and ind == 9:
            w.writerow([word, np.asarray(emo_dict[word]).astype(int)])

    return emo_dict

# gets labels (one hot vec) for emotions and condition
def getLabels(emotion_dict, reddit_text):

    # text_subreddit_file = open(reddit_text)
    w = csv.writer(open("dataset_labelled.csv", "a"))

    with open(reddit_text, newline="", encoding='Latin1') as csvfile:

        # TODO: might need to change this depending on structure of csv
        reader = csv.reader(csvfile) 
        for row in reader:

            #TODO: might need more preprocessing depending on what the csv acutally looks like 
            text = row[0] 
            subreddit = row[1]
            if text: # classes: [depression, anxiety, bipolar, addiction, ADHD]
                if subreddit == 'depression':
                    condition_label = "[1,0,0,0,0,0]"
                elif subreddit == 'Anxiety': #r/Anxiety
                    condition_label = "[0,1,0,0,0,0]"
                elif subreddit == 'BipolarReddit':
                    condition_label = "[0,0,1,0,0,0]"
                elif subreddit == 'addiction':
                    condition_label = "[0,0,0,1,0,0]"
                elif subreddit == 'ADHD':
                    condition_label = "[0,0,0,0,1,0]"
                elif subreddit == "AskReddit" or subreddit == "Showerthoughts" or subreddit == "jokes":
                    condition_label = "[0,0,0,0,0,1]"
                else:
                    print(f"Wrong subreddit? {subreddit}.", )
                    condition_label = None
                
                if condition_label:# write to dataset

                    # process emotions
                    emotion_label = np.zeros((8,)).astype(int)
                    for word in text.split(): 
                        # emotion label: [anger, anticipation, disgust, feat, joy, sadness, surprise, trust]
                        emotion_word = np.asarray(emotion_dict[word]).astype(int) if word in emotion_dict else np.zeros((8,)).astype(int)
                        emotion_label = (emotion_label | emotion_word)
                        emotion_label = list(emotion_label)
                    
                    w.writerow([text.encode('Latin1'), condition_label, emotion_label]) #TODO: add emotion label #' '.join(map(str, a))
    print("done processing, created dataset")
    return 

lexicon_path = "./NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
reddit_text_path = 'D:\CS 224n\cs224n_project\data\\all_raw_comments.csv' # replace with relevant path

# set write_to_disk = True if want to save to file
emotion_dict = getEmotionDict(lexicon_path=lexicon_path) 
getLabels(emotion_dict, reddit_text_path) # outputs to file: dataset.csv