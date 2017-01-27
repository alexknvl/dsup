import sys, errno, os, os.path
import ujson as json
from datetime import *
import easytime
import gzip

def ParseDate(string):
    return datetime.strptime(string, '%a %b %d %H:%M:%S +0000 %Y')

def inside_range(range, x):
    return range[0] <= x and x <= range[1]

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

from recordtype import recordtype
from tabulate import tabulate
Tweet = recordtype('Tweet', ['arg1', 'arg2', 'sid', 'words', 'pos',
                             'neTags', 'date', 'y', 'datetime',
                             'from_negative', 'time_since_tweet',
                             'title', 'timestamp', 'features'])

tweet_datetime_cmp = lambda a,b: cmp(a.datetime, b.datetime)

def tsv_print(file, *args):
    file.write('\t'.join(str(s) for s in args) + '\n')

def ListPRF(predictions, N):
    """ Computes a single precision and recall of a list of predictions """
    tp = 0.0
    fp = 0.0
    fn = 0.0
    result = []

    for p in predictions:
        T = p['pred']
        if p['y'] == 1:
            tp += 1
        elif p['y'] == -1:
            fp += 1

        fn = N - tp

        P = 0
        if tp + fp > 0:
            P = tp / (tp + fp)
        R = 0
        if tp + fn > 0:
            R = tp / (tp + fn)
        F = 0
        if P + R > 0:
            F = 2 * P * R / (P + R)

        result.append((T, P, R, F))
    return result

def MaxF1(predictions, N):
    """ Computes maximum F1 """
    tp = 0.0
    fp = 0.0
    fn = 0.0
    maxF = 0.0

    for p in predictions:
        if p['y'] == 1:
            tp += 1
        elif p['y'] == -1:
            fp += 1

        fn = N - tp

        P = 0
        if tp + fp > 0:
            P = tp / (tp + fp)
        R = 0
        if tp + fn > 0:
            R = tp / (tp + fn)
        F = 0
        if P + R > 0:
            F = 2 * P * R / (P + R)

        if maxF < F:
            maxF = F

    return maxF

def PDNU(data):
    total = 0
    positive = 0
    discarded = 0
    unlabeled = 0
    negative = 0

    for tweet in data:
        if tweet[1] == 1:
            positive += 1
        elif tweet[1] == -1:
            negative += 1
        else:
            discarded += 1

        if tweet[-1].from_negative:
            unlabeled += 1

        total += 1

    P = positive / float(total)
    D = discarded / float(total)
    N = negative / float(total)
    U = unlabeled / float(total)

    return (P, D, N, U)

def read_preprocessed_negative(path, train_range, dev_range,
                               train_tweets, dev_tweets,
                               ep_tweets, labeled_ids):
    nNeg = 0
    idEps = set([])

    for line in gzip.open(path):
        nNeg += 1
        if nNeg % 100000 == 0:
            print "read_preprocessed_negative: %s" % nNeg

        tweet = json.loads(line)

        ts = tweet['created_at']
        dt = datetime.fromtimestamp(ts)
        del tweet['created_at']

        tweet = Tweet(
            # arg1=arg1, arg2=arg2,
            title=tweet['arg1'], from_negative=True,
            time_since_tweet=0.0,
            datetime=dt,
            timestamp=ts,
            y=-1,
            features=None,
            **tweet)

        tweet.neTags = [s[0] for s in tweet.neTags.split(' ')]
        tweet.pos = tweet.pos.split(' ')

        isInTrainRange = inside_range(train_range, tweet.datetime)
        isInDevRange = inside_range(dev_range, tweet.datetime)

        if not tweet.sid in labeled_ids:
            if isInTrainRange or isInDevRange:
                ep = "%s\t%s" % (tweet.arg1, tweet.arg2)
                ep_tweets.setdefault(ep, []).append(tweet)

                if isInTrainRange:
                    train_tweets.append(tweet)
                else:
                    dev_tweets.append(tweet)

    train_tweets.sort(tweet_datetime_cmp)
    dev_tweets.sort(tweet_datetime_cmp)

def read_positive(base_path, attr_group,
                  train_range, dev_range,
                  train_tweets, dev_tweets,
                  ep_tweets, wp_edits,
                  labeled_ids):
    for attr in attr_group:
        print "reading %s" % attr
        idEps = set([])

        for line in open(os.path.join(base_path, attr)):
            (label, rel, arg1, arg2, tweetDate, editDate,
             timeSinceTweet, tweetStr, editStr) = line.strip().split('\t')

            timeSinceTweet = float(timeSinceTweet)

            edit = json.loads(editStr)
            title = edit['title']

            tweet = json.loads(tweetStr)

            del tweet['entity']
            del tweet['eType']
            del tweet['loc']
            del tweet['uid']
            del tweet['eventTags']
            if 'from_date' in tweet:
                del tweet['from_date']

            dt = ParseDate(tweet['created_at'])
            ts = easytime.datetime_to_timestamp(dt, 'utc')
            del tweet['created_at']

            tweet = Tweet(
                arg1=arg1, arg2=arg2,
                title=title, from_negative=False,
                time_since_tweet=timeSinceTweet,
                datetime=dt,
                timestamp=ts,
                y=-1,
                features=None,
                **tweet)

            tweet.neTags = [s[0] for s in tweet.neTags.split(' ')]
            tweet.pos = tweet.pos.split(' ')

            ep  = "%s\t%s" % (arg1,arg2)
            wp_edits.setdefault(ep, set()).add(editStr)

            isInTrainRange = inside_range(train_range, tweet.datetime)
            isInDevRange   = inside_range(dev_range, tweet.datetime)

            if isInTrainRange or isInDevRange:
                if isInTrainRange:
                    train_tweets.append(tweet)
                else:
                    dev_tweets.append(tweet)

                idEp = tweet.sid + '\t' + arg1 + '\t' + arg2
                if idEp in idEps:
                    continue
                idEps.add(idEp)

                ep_tweets.setdefault(ep, []).append(tweet)

                ################################################################################################
                #Keep track of the IDs in the labeled dataset so we can exclude them from the unlabeled data...
                ################################################################################################
                labeled_ids.add(tweet.sid)

    print "sorting train"
    train_tweets.sort(tweet_datetime_cmp)
    print "sorting dev"
    dev_tweets.sort(tweet_datetime_cmp)

