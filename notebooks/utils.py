
import numpy as np
import random
import re
import torch as th

from sklearn.metrics import f1_score


def set_seed_everywhere(seed):
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_f1(eval_pred, thresholds=None, metric="macro"):
    logits, labels = eval_pred
    probs = th.sigmoid(th.from_numpy(logits)).numpy()
    if thresholds is None:
        thresholds = select_thresholds(labels, probs)
    predictions = (probs > thresholds).astype(int)
    return {"f1": f1_score(predictions, labels, average=metric)}


def select_thresholds(eval_labels, eval_probs, search_range=(0.3, 0.7), metric="macro"):
    """Selects the best threshold for each class based on the F1 score."""
    lower, upper = search_range
    assert lower > 0 and upper < 1
    best_thresholds_per_class = []
    for i in range(len(eval_labels.shape[1])):
        candidate_thresholds = np.arange(lower, upper, .01)
        scores = []
        for threshold in candidate_thresholds:
            score = f1_score(
                eval_labels[:, i],
                (eval_probs[:, i] > threshold).astype(int),
                average=metric)
            scores.append(score)
        best_threshold = candidate_thresholds[np.argmax(scores)]
        best_thresholds_per_class.append(best_threshold)
    thresholds = np.array(best_thresholds_per_class)

    return thresholds


# Preprocessing function to clean the tweets.
# Use with caution: removing hashtags and handles _may_ reduce model performance.
def preprocess_tweet(tweet, remove_hashtags=False, remove_handles=False):
    # remove handles, hashtags, urls
    if remove_hashtags:
        tweet = re.sub(r'#\w+', '', tweet)
    if remove_handles:
        tweet = re.sub(r'@\w+', '', tweet)

    # remove urls
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'www\S+', '', tweet)
    tweet = re.sub(r'pic.twitter\S+', '', tweet)

    tweet = re.sub(r'\W', ' ', tweet)  # remove special characters
    tweet = re.sub(r'\s+', ' ', tweet)  # remove multiple whitespaces

    return tweet.strip()
