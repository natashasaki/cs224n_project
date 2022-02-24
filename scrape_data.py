import pandas as pd
from datetime import datetime
from pmaw import PushshiftAPI

def scrape_data():
    subreddit_list = ["addiction", "ADHD", "Anxiety", "AskReddit", "BipolarReddit", "depression", "jokes", "Showerthoughts"]
    for sub in subreddit_list:
        # generate_subreddit_pickles(reddit, subreddit_list)
        subreddit_scraper(sub, 30000)

def subreddit_scraper(subreddit, target):
    api = PushshiftAPI()

    before = int(datetime(2021,12,31,0,0).timestamp())
    after = int(datetime(2020,1,1,0,0).timestamp())
    
    limit=target
    comments = api.search_comments(subreddit=subreddit, limit=limit, before=before, after=after)
    print(f'Retrieved {len(comments)} comments from Pushshift', flush=True)

    comments_df = pd.DataFrame(comments)# preview the comments data
    comments_df.head(5)
    comments_df.to_csv("D:\CS 224n\cs224n_project\data\\" + subreddit + "_comments.csv", header=True, index=False, columns=list(comments_df.axes[1]))

scrape_data()
