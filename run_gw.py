import sys, errno, os, os.path
import LR
import ujson as json
import gzip
from datetime import *
from FeatureExtractor import *
import cProfile
import itertools
import re
import gc
from unidecode import unidecode

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'new'))
import easytime
import fx

sys.path.append('../../weakly_supervised_events/python')

#from HDEM import *
from Classifier import *
from Vocab import *


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
        else:
            raise

from recordtype import recordtype
from tabulate import tabulate

Sample = recordtype('Sample', ['arg1', 'arg2',
                               'doc_id', 'sentence_id', 'sample_id',
                               'date', 'timestamp',
                               'words', 'pos', 'neTags',
                               'randomly_sampled', 'y', 'dt',
                               'features'])


def gw_ne_tags_to_bio(tags):
    result = ['O']
    for t in tags:
        if t != 'O':
            if result[-1] == 'O':
                result.append('B')
            else:
                result.append('I')
        else:
            result.append('O')

    assert len(result) == len(tags) + 1

    return result[1:]


def gw_parse_positive_line(line):
    (attr, arg1, arg2, edit_str, sentence_str) = \
        line.decode('utf-8').strip().split('\t')
    edit_json = json.loads(edit_str)
    sentence_json = json.loads(sentence_str)

    arg1 = unidecode(arg1).lower()
    arg2 = unidecode(arg2).lower()

    #title = edit['title']
    # infobox_name = \
    #     re.sub('[^a-zA-Z]+', '', edit['infobox_name']).lower()

    words = sentence_json['sentence'][1]
    text1 = ' '.join(unidecode(text).lower().replace(' ', '')
                     for i, text, lemma, s, e, pos, ner in words)
    pos1 = [pos for i, text, lemma, s, e, pos, ner in words]
    ner1 = gw_ne_tags_to_bio([ner for i, text, lemma, s, e, pos, ner in words])

    #assert len(words) == len(text1)
    assert len(words) == len(pos1)
    assert len(words) == len(ner1)
    #print ner

    edit_ts = edit_json['timestamp']
    sentence_ts = easytime.ts(*sentence_json['date'])

    doc_id = sentence_json['doc_id']
    sentence_id = sentence_json['sentence'][0]
    ep = "%s\t%s" % (arg1, arg2)
    sample_id = "%s\t%s" % (doc_id, sentence_id)
    full_id = "%s\t%s" % (sample_id, ep)

    sample = Sample(
        arg1=arg1, arg2=arg2,
        date=sentence_json['date'], timestamp=sentence_ts,
        doc_id=doc_id,
        sentence_id=sentence_id,
        sample_id=sample_id,
        words=text1, pos=pos1, neTags=ner1,
        randomly_sampled=False,
        y=-1, dt=edit_ts - sentence_ts,
        features=None)

    return ep, sample_id, full_id, sample, edit_str


def gw_parse_randomly_sampled_file_line(line):
    try:
        _, arg1, arg2, ne1, ne2, sentence_str = \
            line.decode('utf-8').strip().split('\t')
    except:
        print line
        raise
    sentence_json = json.loads(sentence_str)

    arg1 = unidecode(arg1).lower()
    arg2 = unidecode(arg2).lower()

    words = sentence_json['sentence'][1]
    text1 = ' '.join(unidecode(text).lower().replace(' ', '')
                     for i, text, lemma, s, e, pos, ner in words)
    pos1 = [pos for i, text, lemma, s, e, pos, ner in words]
    ner1 = gw_ne_tags_to_bio([ner for i, text, lemma, s, e, pos, ner in words])

    sentence_ts = easytime.ts(*sentence_json['date'])

    doc_id = sentence_json['doc_id']
    sentence_id = sentence_json['sentence'][0]
    ep = "%s\t%s" % (arg1, arg2)
    sample_id = "%s\t%s" % (doc_id, sentence_id)
    full_id = "%s\t%s" % (sample_id, ep)

    sample = Sample(
        arg1=arg1, arg2=arg2,
        date=sentence_json['date'], timestamp=sentence_ts,
        doc_id=doc_id,
        sentence_id=sentence_id,
        sample_id=sample_id,
        words=text1, pos=pos1, neTags=ner1,
        randomly_sampled=True,
        y=-1, dt=0,
        features=None)

    return ep, sample_id, full_id, sample


def vocabulary_resolve(features, vocabulary, index, immutable=False):
    for feature in features:
        i = index.get(feature, -1)
        if i == -1 and not immutable:
            i = len(vocabulary)
            index[feature] = i
            vocabulary.append(feature)
        if i != -1:
            yield i


class GridSearch:
    MIN_FEATURE_COUNT = 2
    MIN_TWEETS = 3

    MODES=["normal", "baseline"]
    XR_MODES=["lr"]
    FEATURE_WINDOW_DAYS=[1]
    EXPECTATIONS=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    NEGATIVE_FILE="../training_data_gw/negative_gw1.gz"
    TRAIN_RANGE=(easytime.ts(year=2004, month=6, day=1),
                 easytime.ts(year=2009, month=6, day=1))
    DEV_RANGE=(easytime.ts(year=2009, month=6, day=5),
               easytime.ts(year=2012, month=4, day=1))

    POSITIVE_RANGE = [-10, 3]
    NEGATIVE_RANGE = [-50, 3]

    def classify_by_timediff(self, time_since_tweet):
        oneDay = 24 * 60 * 60.0
        T = 0
        E = time_since_tweet / oneDay

        if inside_range(self.POSITIVE_RANGE, T - E):
            return 1
        elif not inside_range(self.NEGATIVE_RANGE, T - E):
            return -1
        else:
            return 0

    def classify_tweet(self, mode, tweet, negative, in_test_set, time_since_tweet):
        if negative:
            tweet.y = -1
            return

        if mode == "normal":
            tweet.y = self.classify_by_timediff(time_since_tweet)
        elif mode == "baseline":
            if in_test_set:
                tweet.y = self.classify_by_timediff(time_since_tweet)
            else:
                tweet.y = 1
        else:
            assert False

    def __init__(self, attribute_groups, infoboxes):
        self.attribute_groups = attribute_groups

        self.epTweets    = {}#{}
        self.trainTweets = {}#[]
        self.devTweets   = {}#[]
        self.wpEdits     = {}#{}

        labeledIds = set([])

        datetime_cmp = lambda a, b: cmp(a.timestamp, b.timestamp)

        for attr_group in self.attribute_groups + [['negative']]:
            attr_group_name = ','.join(attr_group)
            self.epTweets[attr_group_name] = {}
            self.trainTweets[attr_group_name] = []
            self.devTweets[attr_group_name] = []
            self.wpEdits[attr_group_name] = {}

        for attr_group in self.attribute_groups:
            attr_group_name = ','.join(attr_group)
            ep_tweets = self.epTweets[attr_group_name]
            train_tweets = self.trainTweets[attr_group_name]
            dev_tweets = self.devTweets[attr_group_name]
            wp_edits = self.wpEdits[attr_group_name]

            for attr in attr_group:
                print "reading %s" % attr
                idEps = set([])
                for line in open("../training_data_gw/" + attr):
                    ep, sample_id, full_id, sample, edit_str = \
                        gw_parse_positive_line(line)

                    if full_id in idEps:
                        continue
                    idEps.add(full_id)

                    ep_tweets.setdefault(ep, []).append(sample)

                    # Keep track of the IDs in the labeled dataset so we can
                    # exclude them from the unlabeled data...
                    labeledIds.add(sample_id)

                    #self.allTweets.append(sample)
                    isInTrainRange = inside_range(self.TRAIN_RANGE,
                                                  sample.timestamp)
                    isInDevRange = inside_range(self.DEV_RANGE,
                                                sample.timestamp)

                    if isInTrainRange:
                        train_tweets.append(sample)
                    elif isInDevRange:
                        dev_tweets.append(sample)

                    wp_edits.setdefault(ep, set()).add(edit_str)

            print "sorting train"
            train_tweets.sort(datetime_cmp)
            print "sorting dev"
            dev_tweets.sort(datetime_cmp)

            print "done reading %s" % attr_group_name
            print "len(trainTweets)=%s" % len(train_tweets)
            print "len(devTweets)=%s" % len(dev_tweets)

        nNeg = 0
        idEps = set([])

        if True:
            attr_group_name = 'negative'
            ep_tweets    = self.epTweets[attr_group_name]
            train_tweets = self.trainTweets[attr_group_name]
            dev_tweets   = self.devTweets[attr_group_name]

            print "reading negative"

            for line in fx.sample(gzip.open(self.NEGATIVE_FILE), rate=10):
                nNeg += 1
                if nNeg % 100000 == 0:
                    print "number of negative read: %s" % nNeg

                try:
                    ep, sample_id, full_id, sample = \
                        gw_parse_randomly_sampled_file_line(line)
                except:
                    continue

                in_train_range = inside_range(self.TRAIN_RANGE,
                                              sample.timestamp)
                in_dev_range = inside_range(self.DEV_RANGE, sample.timestamp)

                if not sample_id in labeledIds:
                    ep_tweets.setdefault(ep, []).append(sample)

                    if in_train_range:
                        train_tweets.append(sample)
                    elif in_dev_range:
                        dev_tweets.append(sample)

            print "sorting train"
            train_tweets.sort(datetime_cmp)
            print "sorting dev"
            dev_tweets.sort(datetime_cmp)

            print "done reading %s" % attr_group_name
            print "len(trainTweets)=%s" % len(train_tweets)
            print "len(devTweets)=%s" % len(dev_tweets)

        table = []
        attrs = list(set(self.trainTweets.keys()) - set(['negative'])) + ['negative']
        for a in attrs:
            table.append((a, len(self.trainTweets[a]), len(self.devTweets[a])))
        print tabulate(table, headers=['attr', 'train', 'dev'])

        # print "Preprocessing/ConsistencyChecking started."
        # for attr_group in self.attribute_groups + [['negative']]:
        #     attr_group_name = ','.join(attr_group)
        #     train_tweets = self.trainTweets[attr_group_name]
        #     dev_tweets = self.devTweets[attr_group_name]
        #     ep_tweets = self.epTweets[attr_group_name]
        #     for tweet in fx.concat(train_tweets, dev_tweets):
        #         assert tweet.features[0] == 0
        #         assert tweet.features is not None
        #     for ep, tweets in ep_tweets.iteritems():
        #         for tweet in tweets:
        #             assert tweet.features[0] == 0
        #             assert tweet.features is not None
        # print "Preprocessing/ConsistencyChecking is done."

    def ListPRF(self, predictions, N):
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

    def MaxF1(self, predictions, N):
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

    def MakePredictions(self, model, devData, wp_edits):
        predictions = []
        for fields in devData:
            x = fields[0]
            y = fields[1]
            prediction = model.Predict(x)
            tweet = fields[-1]
            arg1 = tweet.arg1
            arg2 = tweet.arg2
            ep = "%s\t%s" % (arg1,arg2)
            #print ep
            wp_edits_ep = list(wp_edits.get(ep, []))
            editDate = None
            if len(wp_edits_ep) > 0:
                editDate = datetime.fromtimestamp(json.loads(wp_edits_ep[0])['timestamp'])
            predictions.append({'y':y, 'pred':prediction,
                                'arg1':arg1, 'arg2':arg2,
                                'date':tweet.date,
                                'editDate':editDate,
                                'tweet':tweet , 'wpEdits':wp_edits_ep})

        return predictions

    def Run(self, output_dir):
        print "Preprocessing/ComputeEpFrequencies started."
        ep_counts = {}
        for attr_group in self.attribute_groups + [['negative']]:
            attr_group_name = ','.join(attr_group)
            train_tweets = self.trainTweets[attr_group_name]
            dev_tweets = self.devTweets[attr_group_name]
            for tweet in fx.concat(train_tweets, dev_tweets):
                ep = (tweet.arg1, tweet.arg2)
                ep_counts[ep] = ep_counts.setdefault(ep, 0) + 1
        print "Preprocessing/ComputeEpFrequencies is done."

        print "Preprocessing/DropEps started."
        discard_eps = set()
        for ep, cnt in ep_counts.iteritems():
            if cnt < self.MIN_TWEETS:
                discard_eps.add(ep)
        ep_counts = None

        for attr_group in self.attribute_groups + [['negative']]:
            attr_group_name = ','.join(attr_group)
            self.trainTweets[attr_group_name] = \
                [x for x in self.trainTweets[attr_group_name]
                 if (x.arg1, x.arg2) not in discard_eps]
            self.devTweets[attr_group_name] = \
                [x for x in self.devTweets[attr_group_name]
                 if (x.arg1, x.arg2) not in discard_eps]

            ep_tweets = self.epTweets[attr_group_name]
            for ep in discard_eps:
                ep_str = '%s\t%s' % ep
                if ep_str in ep_tweets:
                    del ep_tweets[ep_str]
        discard_eps = None
        print "Preprocessing/DropEps is done."

        table = []
        attrs = list(set(self.trainTweets.keys()) - set(['negative'])) + ['negative']
        for a in attrs:
            samples_train = len(self.trainTweets[a])
            samples_test = len(self.devTweets[a])
            documents_train = len(set(t.doc_id for t in self.trainTweets[a]))
            documents_test = len(set(t.doc_id for t in self.devTweets[a]))
            table.append((a, samples_train, documents_train, samples_test, documents_test))
        print tabulate(table, headers=['Attr', 'Train Samples', 'Train Documents', 'Test Samples', 'Test Documents'], tablefmt='latex')

        print "Preprocessing/ExtractingFeatures started."
        NO_UNLABELED_FEATURES = True
        vocabulary = []
        index = {}
        vocabulary_resolve(['__BIAS__'], vocabulary, index)

        for attr_group in self.attribute_groups + [['negative']]:
            attr_group_name = ','.join(attr_group)
            train_tweets = self.trainTweets[attr_group_name]
            dev_tweets = self.devTweets[attr_group_name]
            for tweet in fx.concat(train_tweets, dev_tweets):
                features = fx.concat(
                    ['__BIAS__'], generate_binary_features(tweet))
                # features = generate_binary_features(tweet)
                if attr_group_name == 'negative' and NO_UNLABELED_FEATURES:
                    tweet.features = sorted(set(
                        vocabulary_resolve(features, vocabulary, index,
                                           immutable=True)))
                else:
                    tweet.features = sorted(set(
                        vocabulary_resolve(features, vocabulary, index)))
        print "n features: %s" % len(vocabulary)
        print "Preprocessing/ExtractingFeatures is done."

        print "Preprocessing/ComputeFeatureFrequencies started."
        feature_eps = [None] * len(vocabulary)
        for i, _ in enumerate(feature_eps):
            feature_eps[i] = set()

        for attr_group in self.attribute_groups + [['negative']]:
            attr_group_name = ','.join(attr_group)
            train_tweets = self.trainTweets[attr_group_name]
            for tweet in train_tweets:
                for feature in tweet.features:
                    ep = (tweet.arg1, tweet.arg2)
                    feature_eps[feature].add(ep)
        for i, _ in enumerate(feature_eps):
            feature_eps[i] = len(feature_eps[i])
        print "Preprocessing/ComputeFeatureFrequencies is done."

        print "Preprocessing/RemapFeatures started."
        new_vocabulary = []
        new_index = {}
        remapping = [0] * len(vocabulary)

        for i, f in enumerate(vocabulary):
            if feature_eps[i] >= self.MIN_FEATURE_COUNT:
                new_index[f] = len(new_vocabulary)
                remapping[i] = len(new_vocabulary)
                new_vocabulary.append(f)
            else:
                remapping[i] = -1
        index = new_index
        vocabulary = new_vocabulary

        for attr_group in self.attribute_groups + [['negative']]:
            attr_group_name = ','.join(attr_group)
            train_tweets = self.trainTweets[attr_group_name]
            dev_tweets = self.devTweets[attr_group_name]
            for tweet in fx.concat(train_tweets, dev_tweets):
                features = sorted(set(remapping[i] for i in tweet.features))
                if len(features) > 0 and features[0] == -1:
                    features = features[1:]
                tweet.features = features
        print "n features: %s" % len(vocabulary)
        remapping = None
        feature_eps = None
        print "Preprocessing/RemapFeatures is done."

        return

        print "Preprocessing/UnlabeledWindows started."
        unlabeled_training_set = self.GenData(
            tweets=self.trainTweets['negative'],
            feature_window_days=1,
            ep_tweets=self.epTweets['negative'],
            ep_tweets_neg={})
        unlabeled_dev_set = self.GenData(
            tweets=self.devTweets['negative'],
            feature_window_days=1,
            ep_tweets=self.epTweets['negative'],
            ep_tweets_neg={})
        print "len(unlabeled_training_set)=%s" % len(unlabeled_training_set)
        print "len(unlabeled_dev_set)=%s" % len(unlabeled_dev_set)
        print "Preprocessing/UnlabeledWindows is done."

        nsr = 'all'
        self.ClassifyAll("normal")

        last_time = easytime.now()

        for attr_group in self.attribute_groups:
            attr_group_name = ','.join(attr_group)

            ep_tweets     = self.epTweets[attr_group_name]
            ep_tweets_neg = self.epTweets['negative']

            wp_edits     = self.wpEdits[attr_group_name]

            train_tweets = self.trainTweets[attr_group_name]
            dev_tweets = self.devTweets[attr_group_name]

            attr_all_file = open(os.path.join(output_dir, "%s.gs" % attr_group_name), 'w+')

            for fwd in self.FEATURE_WINDOW_DAYS:
                trainData, devData, model, predictions = None, None, None, None
                gc.collect()

                extract_start = easytime.now()
                # trainData, devData = self.ExtractFeatures(
                #     train_tweets, dev_tweets, fwd, ep_tweets, ep_tweets_neg)
                print "Extracting Features"
                trainData = self.GenData(
                    tweets=train_tweets,
                    feature_window_days=1,
                    ep_tweets=ep_tweets,
                    ep_tweets_neg={}) + unlabeled_training_set
                devData = self.GenData(
                    tweets=dev_tweets,
                    feature_window_days=1,
                    ep_tweets=ep_tweets,
                    ep_tweets_neg={}) + unlabeled_dev_set
                print "Done Extracting Features"
                print "Extraction took %s seconds." % (easytime.now() - extract_start)

                for mode, xr in itertools.product(self.MODES, self.XR_MODES):
                    print "Reclassifying"
                    self.ClassifyAll(mode)

                    for v in itertools.chain(iter(trainData), iter(devData)):
                        # [fe.Features(), target.y, tweetDate, title, arg2, target]
                        target = v[-1]
                        v[1] = target.y

                    # P, D, U, N = self.PDU(trainData)
                    # print "P = %.5f" % P
                    # print "D = %.5f" % D
                    # print "U = %.5f" % U
                    # print "N = %.5f" % N

                    model = LR_XRclassifierV2() if xr == "lrxr" else LRclassifierV2()
                    model.Prepare([x for x in trainData if x[1] != 0],
                                   vocabulary, index)

                    for p_ex in self.EXPECTATIONS if xr == "lrxr" else [0]:
                        if xr == "lrxr":
                            model.Train(p_ex=p_ex, l2=100.0)
                        else:
                            model.Train(l2=100.0)

                        predictions = self.MakePredictions(model, devData, wp_edits)

                        N = sum(1 for x in predictions if x['y'] == 1)
                        predictions.sort(lambda a,b: cmp(b['pred'], a['pred']))
                        #F = self.MaxF1(predictions, N)

                        subdir = "%s_%s_%s_%s_%s_%s" % (attr_group_name, mode, xr, fwd, nsr, p_ex)
                        mkdir_p(os.path.join(output_dir, subdir))
                        paramOut   = open(os.path.join(output_dir, subdir, 'paramOut'), 'w+')
                        PRout      = open(os.path.join(output_dir, subdir, 'PRout'), 'w+')
                        maxFout    = open(os.path.join(output_dir, subdir, 'maxFout'), 'w+')
                        predOut    = open(os.path.join(output_dir, subdir, 'predOut'), 'w+')

                        maxF = 0
                        PR = self.ListPRF(predictions, N)
                        for i in range(len(predictions)):
                            (T, P, R, F) = PR[i]
                            PRout.write("%s\t%s\t%s\t%s\n" % (T, P,R,F))
                            if F > maxF:
                                maxF = F
                        PRout.close()

                        done_time = easytime.now()
                        time = done_time - last_time
                        last_time = done_time

                        print "%s/%s/%s/%s: F1=%s in %s s" % (attr_group_name, mode, xr, p_ex, maxF, time)

                        paramOut.write("mode                 = %s\n" % mode)
                        paramOut.write("xr_mode              = %s\n" % xr)
                        paramOut.write("feature_window_days  = %s\n" % fwd)
                        paramOut.write("negative_sample_rate = %s\n" % nsr)
                        paramOut.write("p_ex                 = %s\n" % p_ex)
                        paramOut.write("F                    = %s\n" % maxF)
                        paramOut.write("time                 = %s\n" % time)
                        paramOut.close()

                        maxFout.write(str(maxF) + "\n")
                        maxFout.close()

                        for p in predictions:
                            p['tweet'] = p['tweet']._asdict()
                            del p['editDate']

                            predOut.write(json.dumps(p) + "\n")
                        predOut.close()

                        model.PrintWeights(os.path.join(output_dir, subdir, 'weights'))

                        attr_all_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (attr_group_name, mode, xr, fwd, nsr, p_ex, maxF))
                        attr_all_file.flush()


            attr_all_file.close()

    #PRECONDITION: tweets must be sorted by time before calling GenData
    def GenData(self, tweets, feature_window_days, ep_tweets, ep_tweets_neg):
        result = []

        for target in tweets:
            arg1 = target.arg1
            arg2 = target.arg2
            ep = "%s\t%s" % (arg1, arg2)

            labeled = ep_tweets.get(ep, [])
            unlabeled = ep_tweets_neg.get(ep, [])

            #if len(labeled) + len(unlabeled) >= self.MIN_TWEETS:
            dt = easytime.dt(days=feature_window_days)
            predicate = lambda x: x.timestamp <= target.timestamp and \
                                  x.timestamp > (target.timestamp - dt)
            window = [x.features for x in fx.concat(labeled, unlabeled)
                      if predicate(x)]
            window_features = sorted(set(fx.concat(*window)))
            result.append([window_features, target.y, target])

        return result

    def PDU(self, data):
        total = 0
        positive = 0
        discarded = 0
        negative = 0
        unknown = 0

        for tweet in data:
            if tweet[1] == 1:
                positive += 1
            elif tweet[1] == -1:
                unknown += 1
            else:
                discarded += 1

            if tweet[-1].from_negative:
                negative += 1

            total += 1

        P = positive / float(total)
        D = discarded / float(total)
        U = unknown / float(total)
        N = negative / float(total)

        return (P, D, U, N)

    def ClassifyAll(self, mode):
        all_tweets = fx.concat(*(self.trainTweets.values() + self.devTweets.values()))
        neg_count = 0
        for tweet in all_tweets:
            isInTrainRange = inside_range(self.TRAIN_RANGE, tweet.timestamp)
            isInDevRange = inside_range(self.DEV_RANGE, tweet.timestamp)
            self.classify_tweet(mode, tweet, tweet.randomly_sampled, isInDevRange, tweet.dt)
            if tweet.randomly_sampled:
                tweet.y = -1
                neg_count += 1
        print "neg_count=%s" % neg_count

        # total = 0
        # positive = 0
        # discarded = 0
        # unknown = 0
        # for tweet in fx.concat(*(self.trainTweets.values() + self.devTweets.values())):
        #     if tweet.y == 1:
        #         positive += 1
        #     elif tweet.y == -1:
        #         unknown += 1
        #     else:
        #         discarded += 1
        #     total += 1

        # P = positive / float(total)
        # D = discarded / float(total)
        # U = unknown / float(total)

        # print "%s, %s, %s" % (P, D, U)

    def ExtractFeatures(self, trainTweets, devTweets, feature_window_days,
                        ep_tweets, ep_tweets_neg):
        print "Extracting Features"
        trainData = self.GenData(trainTweets, feature_window_days, ep_tweets, ep_tweets_neg)
        fs = FeatureSelectionEntity(trainTweets, trainData)
        trainData = fs.FilterFeaturesByCount(self.MIN_FEATURE_COUNT)

        devData = self.GenData(devTweets, feature_window_days, ep_tweets, ep_tweets_neg)
        print "Done Extracting Features"
        print "len(self.train)=%s" % len(trainData)
        print "len(self.dev)=%s" % len(devData)

        return (trainData, devData)

PROFILE=False
PARALLEL=False

import multiprocessing as mp

def run(params):
    attr, infoboxes = params
    ds = GridSearch([attr], infoboxes)
    ds.Run(sys.argv[2])

if __name__ == "__main__":

    pr = None
    if PROFILE:
        pr = cProfile.Profile()
        pr.enable()

    attrs = [s.split(',') for s in sys.argv[1].split(';')]
    infoboxes = [
        'infoboxsenator',
        'infoboxcongressman',
        #'infoboxgovernorelect',
        'infoboxpolitician',
        'infoboxltgovernor',
        'infoboxstatesenator',
        'infoboxofficeholder',
        'infoboxstaterepresentative',
        'infoboxcongressionalcandidate',
        #'infoboxcongressmanelect',
        'infoboxmayor',
        'infoboxspeaker',
        'infoboxgovernor',
        'infoboxcongresswoman',
        'infoboxuniversityundergraduate'
    ]

    if PARALLEL:
        pool = mp.Pool(16)
        pool.map(run, [(a, infoboxes) for a in attrs])
    else:
        ds = GridSearch(attrs, infoboxes)
        ds.Run(sys.argv[2])

    if PROFILE:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
