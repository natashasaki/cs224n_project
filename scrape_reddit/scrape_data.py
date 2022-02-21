from turtle import pos
import praw
import pandas as pd
import pickle
from prawcore.exceptions import Forbidden
import numpy as np
import requests
import time
import random
import csv
from dateutil import rrule
from datetime import datetime, timedelta
import itertools

def scrape_data():
    # scrape the data from reddit (if you haven't already)
    client_id = "3k4zIA2zMKGudA"
    client_secret = "xe_FB-JQTN6lJ-O5WQanle3GsyA"
    user_agent = "Sentiment Analysis Experiment"

    reddit = praw.Reddit(client_id=client_id,
                            client_secret=client_secret,
                            user_agent=user_agent)
    
    subreddit_list = ["addiction", "ADHD", "BipolarReddit", "Anxiety", "depression", "AskReddit", "lifeofnorman", "Showerthoughts"]

    # generate_subreddit_pickles(reddit, subreddit_list)
    subreddit_scraper(subreddit_list, 30000)


def submissions_pushshift_praw(reddit, subreddit, start=None, end=None, limit=100000, extra_query=""):
    """
    A simple function that returns a list of PRAW submission objects during a particular period from a defined sub.
    This function serves as a replacement for the now deprecated PRAW `submissions()` method.

    :param subreddit: A subreddit name to fetch submissions from.
    :param start: A Unix time integer. Posts fetched will be AFTER this time. (default: None)
    :param end: A Unix time integer. Posts fetched will be BEFORE this time. (default: None)
    :param limit: There needs to be a defined limit of results (default: 100), or Pushshift will return only 25.
    :param extra_query: A query string is optional. If an extra_query string is not supplied,
                        the function will just grab everything from the defined time period. (default: empty string)

    Submissions are yielded newest first.

    For more information on PRAW, see: https://github.com/praw-dev/praw
    For more information on Pushshift, see: https://github.com/pushshift/api
    """
    matching_praw_submissions = []

    # Default time values if none are defined (credit to u/bboe's PRAW `submissions()` for this section)
    utc_offset = 28800
    now = int(time.time())
    start = max(int(start) + utc_offset if start else 0, 0)
    end = min(int(end) if end else now, now) + utc_offset

    # Format our search link properly.
    search_link = ('https://api.pushshift.io/reddit/submission/search/'
                   '?subreddit={}&after={}&before={}&sort_type=score&sort=asc&limit={}&q={}')
    search_link = search_link.format(subreddit, start, end, limit, extra_query)

    # Get the data from Pushshift as JSON.
    retrieved_data = requests.get(search_link)
    returned_submissions = retrieved_data.json()['data']

    # Iterate over the returned submissions to convert them to PRAW submission objects.
    for submission in returned_submissions:
        # Take the ID, fetch the PRAW submission object, and append to our list
        praw_submission = reddit.submission(id=submission['id'])
        matching_praw_submissions.append(praw_submission)

    # Return all PRAW submissions that were obtained.
    return matching_praw_submissions

def generate_subreddit_pickles(reddit, subreddits):
    for sub in subreddits:
            curr_sub = reddit.subreddit(sub)
            pickle_name = "/home/elizfitz/cs224n_project/data/" + sub + ".p"
            timestamps = []
            for dt in rrule.rrule(rrule.WEEKLY, dtstart=datetime.fromtimestamp(1577865600), until=datetime.fromtimestamp(1641023999)):
                timestamps.append(int(round(dt.timestamp())))

            timestamps_2020_2022 = [1577865600, 1641023999]

            print("Beginning the " + sub + " scraping", flush=True)
            all_posts = []
            for start_time, end_time in list(zip(timestamps, timestamps[1:])):
                print("Starting to scrape from " + str(start_time) + " to " + str(end_time), flush=True)
                try:
                    all_posts += submissions_pushshift_praw(reddit=reddit, subreddit=curr_sub, start=start_time, end=end_time)
                except:
                    print("There was a bad character.", flush=True)
            print(str(len(all_posts)) + " posts collected!", flush=True)
            random.shuffle(all_posts)
            print("Posts randomized!", flush=True)
            pickle.dump(all_posts, open(pickle_name, "wb"))

def subreddit_scraper(subreddits, target):
    comments = []
    subs = []
    for sub in subreddits:
        pickle_name = "/home/elizfitz/cs224n_project/data/" + sub + ".p"
        data = pickle.load(open(pickle_name, "rb"))
        keep_running = True
        c = target
        p = 0
        for post in data:
            print("I got here with " + str(p) + " posts and " + str(target-c) + " comments.", flush=True)
            if keep_running:
                post.comments.replace_more(limit=None)
                for comment in post.comments.list():
                    try:
                        comments.append(comment.body)
                        subs.append(sub)
                        c -= 1
                        if c == 0:
                            keep_running = False       
                            print("All comments collected!", flush=True)
                            break
                    except:
                        print("There was a bad character.", flush=True)
            p += 1

    df = pd.DataFrame({'text':comments, 'subreddit':subs})
    df.to_csv('/home/elizfitz/cs224n_project/data/raw_comments.csv', index=False)

scrape_data()
