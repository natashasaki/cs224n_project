import pandas as pd
from datetime import datetime
from pmaw import PushshiftAPI

def scrape_data():
    subreddit_list = ["addiction", "ADHD", "Anxiety", "AskReddit", "BipolarReddit", "depression", "jokes", "Showerthoughts"]
    # subreddit list for analysis data
    # subreddit_list = ["personalfinance", "legaladvice", "confessions"]
    for sub in subreddit_list:
        # generate_subreddit_pickles(reddit, subreddit_list)
        subreddit_scraper(sub, 1000, posts=True)

def subreddit_scraper(subreddit, target, posts=False):
    api = PushshiftAPI()

    # first half of analysis time periods
    time_periods =  [(int(datetime(2020,3,31,0,0).timestamp()), int(datetime(2020,1,1,0,0).timestamp())),
                    (int(datetime(2020,6,30,0,0).timestamp()), int(datetime(2020,4,1,0,0).timestamp())),
                    (int(datetime(2020,9,30,0,0).timestamp()), int(datetime(2020,7,1,0,0).timestamp())),
                    (int(datetime(2020,12,31,0,0).timestamp()), int(datetime(2020,10,1,0,0).timestamp()))]

    # second half of analysis time periods
    # time_periods =  [(int(datetime(2021,3,31,0,0).timestamp()), int(datetime(2021,1,1,0,0).timestamp())),
    #                 (int(datetime(2021,6,30,0,0).timestamp()), int(datetime(2021,4,1,0,0).timestamp())),
    #                 (int(datetime(2021,9,30,0,0).timestamp()), int(datetime(2021,7,1,0,0).timestamp())),
    #                 (int(datetime(2021,12,31,0,0).timestamp()), int(datetime(2021,10,1,0,0).timestamp()))]
    frames = []
    for tp in time_periods:
        print(tp, flush=True)
        limit=target
        if not posts:
            comments = api.search_comments(subreddit=subreddit, limit=limit, before=tp[0], after=tp[1])
            print(f'Retrieved {len(comments)} comments from Pushshift', flush=True)
            comments_df = pd.DataFrame(comments)
            frames.append(comments_df.copy())
        else:
            submissions = api.search_submissions(subreddit=subreddit, limit=limit, before=tp[0], after=tp[1], filter=['selftext'])
            print(f'Retrieved {len(submissions)} submissions from Pushshift', flush=True)

            comments_df = pd.DataFrame(submissions)
            frames.append(comments_df.copy())

    result = pd.concat(frames)
    if not posts:
        filename = "D:\CS 224n\cs224n_project\data\\comments_analysis_" + subreddit + "_1.csv"
    else:
        filename = "D:\CS 224n\cs224n_project\data\\posts_analysis_" + subreddit + "_3.csv"
    result.to_csv(filename, header=True, index=False, columns=list(comments_df.axes[1]))

scrape_data()
