import praw
import pandas as pd
from praw.models import MoreComments

# client id
client_id = "eR2Z7WArbpu3lDhXm-B52g"
secret = "fzTH8aRwdn1_NYowXO7Z3yHmug9oCw"
user_agent = "cs224n"

reddit = praw.Reddit(client_id=client_id, client_secret=secret, user_agent=user_agent)

# get 10 hot posts from the MachineLearning subreddit
hot_posts = reddit.subreddit('MachineLearning').hot(limit=10)
for post in hot_posts:
    print(post.title)

posts = []
ml_subreddit = reddit.subreddit('MachineLearning')
for post in ml_subreddit.hot(limit=10):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])
print(posts)

submission = reddit.submission(url="https://www.reddit.com/r/MapPorn/comments/a3p0uq/an_image_of_gps_tracking_of_multiple_wolves_in/")

submission.comments.replace_more(limit=0)
for top_level_comment in submission.comments:
    print(top_level_comment.body)

submission.comments.replace_more(limit=0)
for comment in submission.comments.list():
    print(comment.body)

import random
n_per_subreddit = 30000
conditions = ["depression", "addiction", "ADHD", "Bipolar", "Anxiety", "Other"]
for condition in conditions:
    subreddit = reddit.subreddit(condition)
    data = []
    count = 0 

    while count < n_per_subreddit: # keep scraping
        subreddit_comments = [] # pull into this

        # append to current list of comments of particular condition
        data = data + random.sample(subreddit_comments, 100)
    
    data.to_csv(f"{condition}.csv", index=True)
