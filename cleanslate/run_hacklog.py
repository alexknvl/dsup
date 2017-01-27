#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import division

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import scipy.sparse

import argparse
import ujson as json
import itertools
import gc

import fx
import easytime

import lrxr
from lrxr import model1_obj, model1_grad, Model1Parameters
from lrxr import model2_obj, model2_grad, Model2Parameters
from lrxr import model3_obj, model3_grad, Model3Parameters
from lrxr import model4_obj, model4_grad, Model4Parameters
from lrxr import model5_obj, model5_grad, Model5Parameters
from lrxr import lrxrta5_obj, lrxrta5_grad
from lrxr import model6_obj, model6_grad, Model6Parameters
from lrxr import model7_obj, model7_grad, Model7Parameters
from lrxr import model8_obj, model8_grad, Model8Parameters
from lrxr import model9_obj, model9_grad, Model9Parameters
from lrxr import model10_obj, model10_grad, Model10Parameters
from lrxr import model11_obj, model11_grad, Model11Parameters
from lrxr import XR
from lrxr.distribution import Beta, InvGamma, Gamma, ExpMixture, Normal
from lrxr.distribution import invgamma_from_mean_var, gamma_from_mean_var
from lrxr.distribution import normal_estimate_weighted, normal_pdf
from lrxr.distribution import expmixture_estimate_weighted, distr_scale

from FeatureExtractor import generate_binary_features

from common import Text, Sample, Edit, gen_tprf, mkdir_p, sha1_structure
from vocab import Vocabulary
import data
import twitter as tw
import gigaword as gw

from collections import namedtuple
from tabulate import tabulate

def spliti(array, sizes):
    """Splits array into smaller arrays with given sizes."""
    array = np.asarray(array)
    sizes = np.asarray(sizes, dtype=np.int32)
    assert array.shape[0] == np.sum(sizes)

    ends = np.cumsum(sizes)
    starts = ends - sizes
    result = np.split(array, ends)
    assert len(result) == len(sizes) + 1
    assert len(result[-1]) == 0
    result = result[:-1]

    if isinstance(sizes, tuple):
        return tuple(indices), tuple(result)
    else:
        return starts, result


def split(array, sizes):
    """Splits array into smaller arrays with given sizes."""
    indices, subarrays = spliti(array, sizes)
    return subarrays


def subsum(array, sizes):
    return np.array([np.sum(x, axis=0) for x in split(array, sizes)])


def submax(array, sizes):
    return np.array([np.max(x, axis=0) for x in split(array, sizes)])


def subargmax(array, sizes):
    indices, subarrays = spliti(array, sizes)
    return np.array([indices[i] + np.argmax(x, axis=0)
                     for i, x in enumerate(subarrays)])


def submode(x, w, sizes):
    x = np.asarray(x)
    w = np.asarray(w)

    indices = subargmax(w, sizes)
    return x[indices]


def meanvar(x, w):
    x = np.asarray(x)
    w = np.asarray(w)
    w = w / w.sum()
    mu = (x * w).sum()
    s2 = (x ** 2 * w).sum() - mu**2
    return mu, s2


def submeanvar(x, w, sizes):
    x = np.asarray(x)
    w = np.asarray(w)
    wn = subsum(w, sizes)
    Ex = subsum(x * w, sizes) / wn
    Ex2 = subsum(x**2 * w, sizes) / wn
    return Ex, Ex2 - Ex**2


def tsv_print(file, *args):
    file.write('\t'.join(str(s) for s in args) + '\n')


def read_preprocessed_negative(gigaword, path, train_range, dev_range,
                               train_tweets, dev_tweets,
                               ep_tweets, labeled_ids):
    nNeg = 0

    samples = gw.read_unmatched(path) if gigaword else tw.read_unmatched(path)

    for s in samples:
        if s is None:
            continue
        nNeg += 1
        if nNeg % 100000 == 0:
            print("read_preprocessed_negative: %s" % nNeg)

        isInTrainRange = train_range[0] <= s.timestamp <= train_range[1]
        isInDevRange = dev_range[0] <= s.timestamp <= dev_range[1]

        # if hash(s.args) % 13 != 0:
        #     continue

        if not s.id in labeled_ids:
            if isInTrainRange or isInDevRange:
                ep_tweets.setdefault(s.args, []).append(s)

                if isInTrainRange:
                    train_tweets.append(s)
                else:
                    dev_tweets.append(s)

    train_tweets.sort(key=lambda s: s.timestamp)
    dev_tweets.sort(key=lambda s: s.timestamp)


def read_positive(gigaword, base_path, attributes,
                  train_range, dev_range,
                  train_tweets, dev_tweets,
                  ep_tweets, labeled_ids):
    for attr in attributes:
        print("  Reading wiki attribute: %s" % attr.name)
        idEps = set([])

        path = os.path.join(base_path, attr.name)

        samples = gw.read_matched(path) if gigaword else \
            tw.read_matched(path)

        for s in samples:
            if s is None:
                continue

            isInTrainRange = train_range[0] <= s.timestamp <= train_range[1]
            isInDevRange = dev_range[0] <= s.timestamp <= dev_range[1]

            if attr.infoboxes is not None:
                goodInfobox = any(e.relation.split('/')[0] in attr.infoboxes
                                  for e in s.edits)
            else:
                goodInfobox = True

            # if hash(s.args) % 13 != 0:
            #     continue

            if (isInTrainRange or isInDevRange) and goodInfobox:
                samples = []
                invertedArgs = tuple(reversed(s.args))
                if attr.invert:
                    s = Sample(s.id, s.timestamp, s.text,
                               invertedArgs, s.edits, -1, None)

                samples.append(s)
                if attr.symmetric:
                    samples.append(
                        Sample(s.id, s.timestamp, s.text,
                               invertedArgs, s.edits, -1, None))

                for s in samples:
                    idEp = (s.id,) + s.args
                    if idEp in idEps:
                        continue
                    idEps.add(idEp)

                    if isInTrainRange:
                        train_tweets.append(s)
                    else:
                        dev_tweets.append(s)

                    ep_tweets.setdefault(s.args, []).append(s)

                    # Keep track of the IDs in the labeled dataset so
                    # we can exclude them from the unlabeled data...
                    labeled_ids.add(s.id)

    train_tweets.sort(key=lambda s: s.timestamp)
    dev_tweets.sort(key=lambda s: s.timestamp)


def read_data(gigaword, base_dir, unmatched_file_name, event_mapping,
              train_range, test_range):
    epTweets = {}
    trainTweets = {}
    devTweets = {}
    labeled_ids = set()

    unlabeled = data.EventMapping("Unlabeled", None)

    for mapping in event_mapping + [unlabeled]:
        epTweets[mapping.name] = {}
        trainTweets[mapping.name] = []
        devTweets[mapping.name] = []

    for mapping in event_mapping:
        ep_tweets = epTweets[mapping.name]
        train_tweets = trainTweets[mapping.name]
        dev_tweets = devTweets[mapping.name]

        print("Reading %s" % mapping.name)

        read_positive(
            gigaword,
            base_dir, mapping.attributes,
            train_range, test_range,
            train_tweets, dev_tweets,
            ep_tweets, labeled_ids)

        print("  Done reading %s" % mapping.name)
        print("  len(trainTweets)=%s" % len(train_tweets))
        print("  len(devTweets)=%s" % len(dev_tweets))

    for mapping in [unlabeled]:
        ep_tweets = epTweets[mapping.name]
        train_tweets = trainTweets[mapping.name]
        dev_tweets = devTweets[mapping.name]

        print("reading negative")

        read_preprocessed_negative(
            gigaword,
            unmatched_file_name,
            train_range, test_range,
            train_tweets, dev_tweets,
            ep_tweets, labeled_ids)

        print("done reading %s" % mapping.name)
        print("len(trainTweets)=%s" % len(train_tweets))
        print("len(devTweets)=%s" % len(dev_tweets))

    return epTweets, trainTweets, devTweets


def np_classify_by_timediff(p, n, dTE):
    in_positive = np.logical_and(p[0] <= dTE, dTE <= p[1])
    in_negative = np.logical_and(n[0] <= dTE, dTE <= n[1])
    return np.where(in_positive, 1, np.where(in_negative, 0, -1))


def np_classify_normal(p, n, dTE, is_unmatched, in_test_set):
    # Notice that we call nan_to_num here, since the only case
    # where we can get a zero as dTE is when there are no edits
    # (i.e. an unmatched sample), but this case is already handled
    # by the where(is_unmatched, ...) expression.
    by_timediff = np_classify_by_timediff(p, n, np.nan_to_num(dTE))
    # print(by_timediff.shape)
    # print(is_unmatched.shape)
    return np.where(is_unmatched, -1, by_timediff)


def np_classify_baseline(p, n, dTE, is_unmatched, in_test_set):
    # Notice that we call nan_to_num here, since the only case
    # where we can get a zero as dTE is when there are no edits
    # (i.e. an unmatched sample), but this case is already handled
    # by the where(is_unmatched, ...) expression.
    return np.where(
        is_unmatched, -1,
        np.where(
            in_test_set, np_classify_by_timediff(p, n, np.nan_to_num(dTE)),
            1))


Prediction = namedtuple('Prediction', ['y', 'p', 'sample'])


def make_predictions(y, p, data):
    predictions = []
    for i, fields in enumerate(data):
        s = fields[-1]
        predictions.append(Prediction(y=y[i], p=p[i], sample=s))
    return predictions


def generate_data(tweets, feature_window_days, ep_tweets, ep_tweets_neg,
                  only_ids=None):
    result = []

    for s in tweets:
        if only_ids is not None:
            if s.id not in only_ids:
                continue
        labeled = ep_tweets.get(s.args, [])
        unlabeled = ep_tweets_neg.get(s.args, [])

        dt = easytime.dt(days=feature_window_days)
        predicate = lambda x: x.timestamp <= s.timestamp and \
                              x.timestamp > (s.timestamp - dt)
        window = [x for x in fx.concat(labeled, unlabeled)
                  if predicate(x)]

        window = [x.features for x in window]

        window_features = sorted(set(fx.concat(*window)))
        result.append([window_features, s.y, s])

    return sorted(result, key=lambda r: r[-1].args)


def make_sparse_matrix(rows, column_count):
    x = scipy.sparse.lil_matrix((len(rows), column_count))
    for i, js in enumerate(rows):
        for j in js:
            x[i, j] = 1
    return x.tocsr()


def make_data(data, vocabulary):
    acnt = []
    arg_voc = Vocabulary()
    mdata = (v for v in data if len(v[-1].edits) != 0)
    for i, (k, g) in enumerate(fx.run_group_by(mdata, lambda v: v[-1].args)):
        assert i == arg_voc.update(k)
        acnt.append(len(list(g)))
    acnt = np.array(acnt, dtype=np.int)

    m = np.array([True if len(v[-1].edits) != 0 else False for v in data])

    sample_times = np.array([v[-1].timestamp for v in data])
    edit_times = np.array([v[-1].edits[0].timestamp
                           if len(v[-1].edits) != 0
                           else np.nan
                           for v in data])
    dt = (edit_times - sample_times) / easytime.dt(days=1)

    ux = make_sparse_matrix([v[0] for v in data], len(vocabulary))
    x = make_sparse_matrix([v[0] for v in data if len(v[-1].edits) != 0],
                           len(vocabulary))
    u = make_sparse_matrix([v[0] for v in data if len(v[-1].edits) == 0],
                           len(vocabulary))

    # dt      :  L ⊕ U
    # ux      : (L ⊕ U) ⊗ F
    # x       :  L      ⊗ F
    # u       :      U  ⊗ F
    # matched :  L ⊕ U
    return arg_voc, acnt, dt, x, u, ux, m, sample_times, edit_times


def average_or_zero(p, w):
    old_errors = np.seterr(all='ignore')
    result = np.nan_to_num(np.average(p, weights=w))
    np.seterr(**old_errors)
    return result


def run_models(gigaword,
               base_dir, unmatched_file_name, output_dir,
               event_mapping, train_range, test_range,
               min_tweets_per_entity_pair,
               min_entity_pairs_per_feature,
               modes,
               feature_window_days,
               ha_pos_range, ha_neg_range,
               heval_results,
               annotated_ids,
               well_aligned):
    epTweets, trainTweets, devTweets = read_data(
        gigaword=gigaword,
        base_dir=base_dir,
        unmatched_file_name=unmatched_file_name,
        event_mapping=event_mapping,
        train_range=train_range,
        test_range=test_range)

    # print "!! ", any(s.id == "102053678478925824" for m in devTweets for s in devTweets[m])

    unlabeled_mapping = data.EventMapping("Unlabeled", None)

    table = []
    attrs = list(set(trainTweets.keys()) - set(['Unlabeled'])) + \
        ['Unlabeled']
    for a in attrs:
        train_eps = set(s.args for s in trainTweets[a])
        train_eps = len(train_eps)
        table.append((a, len(trainTweets[a]), len(devTweets[a]), train_eps))
    print(tabulate(table, headers=['attr', 'train', 'dev', 'eps']))

    found_ids = set()
    for mapping in event_mapping + [unlabeled_mapping]:
        dev_tweets = devTweets[mapping.name]
        for s in dev_tweets:
            if s.id in annotated_ids:
                found_ids.add(s.id)
    print "annotated check: ", len(found_ids), len(annotated_ids)

    print("Preprocessing/ComputeEpFrequencies started.")
    ep_counts = {}
    annotated_eps = {}
    for mapping in event_mapping + [unlabeled_mapping]:
        train_tweets = trainTweets[mapping.name]
        dev_tweets = devTweets[mapping.name]
        for s in fx.concat(train_tweets, dev_tweets):
            if s.id in annotated_ids:
                annotated_eps\
                    .setdefault(mapping.name, set([]))\
                    .add(s.args)
            ep_counts[s.args] = ep_counts.setdefault(s.args, 0) + 1
    print("Preprocessing/ComputeEpFrequencies is done.")

    print("Preprocessing/DropEps started.")
    discard_eps = set()
    for ep, cnt in ep_counts.iteritems():
        if cnt < min_tweets_per_entity_pair:
            discard_eps.add(ep)
    ep_counts = None

    for mapping in event_mapping + [unlabeled_mapping]:
        trainTweets[mapping.name] = \
            [s for s in trainTweets[mapping.name]
             if s.args not in discard_eps
             or (mapping.name in annotated_eps and
                 s.args in annotated_eps[mapping.name])]
        devTweets[mapping.name] = \
            [s for s in devTweets[mapping.name]
             if s.args not in discard_eps
             or (mapping.name in annotated_eps and
                 s.args in annotated_eps[mapping.name])]

        ep_tweets = epTweets[mapping.name]
        for ep in discard_eps:
            if mapping.name in annotated_eps and ep in annotated_eps[mapping.name]:
                continue
            if ep in ep_tweets:
                del ep_tweets[ep]
    discard_eps = None
    print("Preprocessing/DropEps is done.")

    # print "!! ", any(s.id == "102053678478925824" for m in devTweets for s in devTweets[m])

    found_ids = set()
    for mapping in event_mapping + [unlabeled_mapping]:
        dev_tweets = devTweets[mapping.name]
        for s in dev_tweets:
            if s.id in annotated_ids:
                found_ids.add(s.id)
    print "annotated check: ", len(found_ids), len(annotated_ids)

    table = []
    attrs = list(set(trainTweets.keys()) - set(['Unlabeled'])) + \
        ['Unlabeled']
    for a in attrs:
        train_eps = set(s.args for s in trainTweets[a])
        train_eps = len(train_eps)
        table.append((a, len(trainTweets[a]), len(devTweets[a]), train_eps))
    print(tabulate(table, headers=['attr', 'train', 'dev', 'eps']))

    print("Preprocessing/ExtractingFeatures started.")
    NO_UNLABELED_FEATURES = True
    vocabulary = Vocabulary(['__BIAS__'])

    for mapping in event_mapping + [unlabeled_mapping]:
        train_tweets = trainTweets[mapping.name]
        dev_tweets = devTweets[mapping.name]
        for s in fx.concat(train_tweets, dev_tweets):
            # args, words, pos, neTags
            features = fx.concat(
                ['__BIAS__'],
                generate_binary_features(s.args,
                                         s.text.words,
                                         s.text.pos,
                                         s.text.ner))
            if mapping.name == 'Unlabeled' and NO_UNLABELED_FEATURES:
                s.features = sorted(set(vocabulary.get(f)
                                        for f in features))
            else:
                s.features = sorted(set(vocabulary.update(f)
                                        for f in features))
    print("n features: %s" % len(vocabulary))
    print("Preprocessing/ExtractingFeatures is done.")

    print("Preprocessing/ComputeFeatureFrequencies started.")
    feature_eps = [None] * len(vocabulary)
    for i, _ in enumerate(feature_eps):
        feature_eps[i] = set()

    for mapping in event_mapping + [unlabeled_mapping]:
        train_tweets = trainTweets[mapping.name]
        for s in train_tweets:
            for feature in s.features:
                feature_eps[feature].add(s.args)
    for i, _ in enumerate(feature_eps):
        feature_eps[i] = len(feature_eps[i])
    print("Preprocessing/ComputeFeatureFrequencies is done.")

    print("Preprocessing/RemapFeatures started.")
    drop_feature_max = max(feature_eps)
    remapping = vocabulary.compress(
        lambda i, w: feature_eps[i] >= min_entity_pairs_per_feature and
        (feature_eps[i] <= drop_feature_max or i == 0))

    for mapping in event_mapping + [unlabeled_mapping]:
        train_tweets = trainTweets[mapping.name]
        dev_tweets = devTweets[mapping.name]
        for s in fx.concat(train_tweets, dev_tweets):
            s.features = [remapping[i] for i in s.features
                          if remapping[i] != -1]

    print("n features: %s" % len(vocabulary))
    remapping = None
    feature_eps = None
    print("Preprocessing/RemapFeatures is done.")

    print("Preprocessing/UnlabeledWindows started.")
    heval_full_ids = set(r.sid for r in heval_results)

    unlabeled_training_set = generate_data(
        tweets=trainTweets['Unlabeled'],
        feature_window_days=1,
        ep_tweets=epTweets['Unlabeled'],
        ep_tweets_neg={})
    unlabeled_dev_set = generate_data(
        tweets=devTweets['Unlabeled'],
        feature_window_days=1,
        ep_tweets=epTweets['Unlabeled'],
        ep_tweets_neg={})
    print("len(unlabeled_training_set)=%s" % len(unlabeled_training_set))
    print("len(unlabeled_dev_set)=%s" % len(unlabeled_dev_set))
    print("Preprocessing/UnlabeledWindows is done.")

    # print("Preprocessing/ConsistencyChecking started.")
    # for mapping in event_mapping + [unlabeled_mapping]:
    #     train_tweets = trainTweets[mapping.name]
    #     dev_tweets = devTweets[mapping.name]
    #     ep_tweets = epTweets[mapping.name]
    #     for s in fx.concat(train_tweets, dev_tweets):
    #         assert s.features[0] == 0
    #         assert s.features is not None
    #     for ep, tweets in ep_tweets.iteritems():
    #         for s in tweets:
    #             assert s.features[0] == 0
    #             assert s.features is not None
    # print("Preprocessing/ConsistencyChecking is done.")

    np.seterr(all='raise', under='ignore')

    rng = np.random.RandomState(14523123)
    init_w0 = rng.randn(len(vocabulary))
    init_w0[0] = 0

    last_time = easytime.now()

    for mapping in event_mapping:
        # done = [
        #     'BandMember', 'HQ', 'Awarded', 'DeathPlace',
        #     'TourBy', 'SocialEventIn', 'CollaborationWith',
        #     'Starring', 'OldLeaderName', 'Spouse', 'OrganizationLeader',
        #     'Champions', 'Successor', 'Television', 'KeyPeople',
        #     'SucceedingTechnology', 'CurrentTeam', 'NewProduct',
        #     'GolfTournamentIn', 'BookBy', 'StateRepresentative']
        #
        # if mapping.name in done:
        #     continue

        ep_tweets = epTweets[mapping.name]
        train_tweets = trainTweets[mapping.name]
        dev_tweets = devTweets[mapping.name]

        if len(train_tweets) == 0 or len(dev_tweets) == 0:
            continue

        for fwd in feature_window_days:
            trainData, devData, model, predictions = None, None, None, None
            gc.collect()

            extract_start = easytime.now()
            print("Labeled Windows started.")
            trainData = generate_data(
                tweets=train_tweets,
                feature_window_days=fwd,
                ep_tweets=ep_tweets,
                ep_tweets_neg={}) + unlabeled_training_set
            devData = generate_data(
                tweets=dev_tweets,
                feature_window_days=fwd,
                ep_tweets=ep_tweets,
                ep_tweets_neg={}) + unlabeled_dev_set

            print "!! ", any(s[-1].id == "102053678478925824" for s in devData)

            print("Labeled Windows is done (%s seconds)." %
                  (easytime.now() - extract_start))

            (train_arg_voc, train_acnt, train_dt,
             train_x, train_u, train_ux, train_matched,
             train_st, train_et) = make_data(trainData, vocabulary)

            (test_arg_voc, test_acnt, test_dt,
             test_x, test_u, test_ux, test_matched,
             test_st, test_et) = make_data(devData, vocabulary)

            all_times = np.array(
                [v[-1].timestamp / easytime.dt(days=1)
                 for v in trainData])
            tau0 = np.max(all_times) - np.min(all_times)

            print "dt      :  L ⊕ U      : %s" % (train_dt.shape,)
            print "ux      : (L ⊕ U) ⊗ F : %s" % (train_ux.shape,)
            print "x       :  L      ⊗ F : %s" % (train_x.shape,)
            print "u       :      U  ⊗ F : %s" % (train_u.shape,)
            print "matched :  L ⊕ U      : %s" % (train_matched.shape,)

            print len(train_arg_voc)

            print "Pretraining"

            train_hy = np.squeeze(np_classify_normal(
                ha_pos_range, ha_neg_range,
                -train_dt, 1 - train_matched,
                False))
            test_hy = np.squeeze(np_classify_normal(
                ha_pos_range, ha_neg_range,
                -test_dt, 1 - test_matched,
                True))

            # λ2 regularization coefficient.
            bias_l2 = 10.0
            feature_l2 = 100.0
            l2 = np.ones_like(init_w0)
            l2 *= feature_l2
            l2[0] = bias_l2

            # Expectation regularization.
            xr_params = XR(
                probability=0.01,
                coefficient=10.0 * train_x.shape[0])

            nonzero = train_hy[train_matched] != 0
            print init_w0.shape
            print train_ux[nonzero].shape
            print train_hy[train_matched][nonzero].shape
            print train_u.shape
            print l2.shape
            (init_w, init_nll, init_status) = lrxr.lrxr_train(
                w=init_w0,
                x=train_ux[nonzero],
                y=train_hy[train_matched][nonzero],
                u=train_u,
                l2=l2,
                t=1.0, p_ex=xr_params.probability,
                xr=xr_params.coefficient, tol=1e-6)
            print "NLL (lrxr): %s" % init_nll

            init_p = lrxr.lr_probability(init_w, train_x)
            init_mu, init_s2 = submeanvar(train_dt[train_matched], init_p, train_acnt)

            init_s2 = np.where(init_s2 <= 0.05 ** 2, 0.05**2, init_s2)

            init_p_ep = np.array(
                [average_or_zero(p, normal_pdf(t, init_mu[i], init_s2[i]))
                 for i, (p, t) in enumerate(zip(
                    split(init_p, train_acnt),
                    split(train_dt[train_matched], train_acnt)))])

            # A good cluster has:
            #  * variance            >= 0.1 ** 2
            #  * variance            <= 10.0 ** 2
            #  * mean                >= -5.0
            #  * weighted probablity >= 0.1
            good_variance = np.logical_and(
                init_s2 > 0.1 ** 2,
                init_s2 < 10.0 ** 2)
            good_mean = init_mu >= -5.0
            good_probability = init_p_ep >= 0.01
            good_cluster = good_variance & good_mean & good_probability
            print "Total pairs: %s" % train_acnt.shape[0]
            print "Good pairs : %s" % np.sum(good_cluster)

            init_mu = np.where(good_cluster, init_mu, 0.0)
            init_s2 = np.where(good_cluster, init_s2, 10.0**2)

            if np.sum(good_cluster) > 0:
                init_Mu, init_S2 = normal_estimate_weighted(
                    init_mu[good_cluster], init_p_ep[good_cluster])
                init_pi, init_lp, init_ln = expmixture_estimate_weighted(
                    init_mu[good_cluster], init_p_ep[good_cluster])
            else:
                init_Mu = 100.0
                init_S2 = 100.0 ** 2
                init_pi = 0.99
                init_lp = 100.0
                init_ln = 1.0

            init_delta = np.sqrt(init_s2)
            init_lam = 1 / init_delta

            if init_pi == 1.0:
                init_pi = 0.99

            init_Mu = np.asarray(init_Mu)
            init_S2 = np.asarray(init_S2)
            init_pi = np.asarray(init_pi)
            init_lp = np.asarray(init_lp)
            init_ln = np.asarray(init_ln)

            print "init_Mu       : ", init_Mu
            print "sqrt(init_S2) : ", np.sqrt(init_S2)
            print "init_pi       : ", init_pi
            print "1 / init_lp   : ", 1 / init_lp
            print "1 / init_ln   : ", 1 / init_ln
            print "sqrt(init_s2) : ", np.sqrt(np.mean(init_s2))
            print "1 / init_lam  : ", np.mean(1 / init_lam)

            print "Pretraining done"

            for mode in modes:
                gc.collect()

                if mode == 'baseline':
                    train_hy = np.squeeze(np_classify_baseline(
                        ha_pos_range, ha_neg_range,
                        -train_dt, 1 - train_matched,
                        False))
                    test_hy = np.squeeze(np_classify_baseline(
                        ha_pos_range, ha_neg_range,
                        -test_dt, 1 - test_matched,
                        True))
                else:
                    train_hy = np.squeeze(np_classify_normal(
                        ha_pos_range, ha_neg_range,
                        -train_dt, 1 - train_matched,
                        False))
                    test_hy = np.squeeze(np_classify_normal(
                        ha_pos_range, ha_neg_range,
                        -test_dt, 1 - test_matched,
                        True))

                print "hy      : L ⊕ U       : %s" % (train_hy.shape,)

                if True:
                    Mu_params = Normal(100, 125.0**2)
                    mu_normal_params = Normal(100.0, 125.0**2)
                    mu_expmixture_params = ExpMixture(
                        pi=Beta(3.0, 1.0),
                        lp=Gamma(1.0, 100.0),
                        ln=Gamma(1.0, 100.0))
                    lam_params = Gamma(1.0, 1.0)
                    s2_params = invgamma_from_mean_var(5.0, 500.0**2)

                    if mode == "ad-hoc" or mode == "baseline":
                        parameters = {
                            'attr': mapping.name,
                            'mode': mode,
                            'bias_l2': bias_l2,
                            'feature_l2': feature_l2,
                            'xr': dict(xr_params._asdict())
                        }

                        nonzero = train_hy[train_matched] != 0
                        print nonzero.shape
                        (w, nll, status) = lrxr.lrxr_train(
                            w=init_w0,
                            x=train_ux[nonzero],
                            y=train_hy[train_matched][nonzero],
                            u=train_u,
                            l2=l2,
                            t=1.0,
                            p_ex=xr_params.probability,
                            xr=xr_params.coefficient,
                            tol=1e-6)
                        del status['grad']
                        print(status)
                        print(nll)

                        predictions = make_predictions(test_hy, lrxr.lr_probability(w, test_ux), devData)
                        train_predictions = make_predictions(train_hy, lrxr.lr_probability(w, train_ux), trainData)
                        weights = sorted(zip(range(len(vocabulary)), vocabulary.words, w),
                                         key=lambda t: -t[-1])
                    elif mode == "model10":
                        parameters = {
                            'attr': mapping.name,
                            'mode': mode,
                            'bias_l2': bias_l2,
                            'feature_l2': feature_l2,
                            'xr': dict(xr_params._asdict()),
                            'Mu': dict(Mu_params._asdict()) }

                        min_mu = np.ones_like(init_mu) * -5
                        max_mu = np.ones_like(init_mu) * np.inf
                        min_s2 = np.ones_like(init_s2) * 0.1 ** 2
                        max_s2 = np.ones_like(init_s2) * 10.0 ** 2
                        min_w = np.ones_like(init_w) * -np.inf
                        max_w = np.ones_like(init_w) * np.inf

                        (w, nll, status) = lrxr.train_adadelta(
                            model10_obj, model10_grad,
                            w0=np.concatenate(([init_Mu, init_S2], init_mu, init_s2, init_w)),
                            wmin=np.concatenate(([0.0, 0.1], min_mu, min_s2, min_w)),
                            wmax=np.concatenate(([np.inf, np.inf], max_mu, max_s2, max_w)),
                            parameters=Model10Parameters(
                                tau=tau0,
                                Mu=Mu_params,
                                s2=s2_params,
                                l2=l2, t=1.0, xr=xr_params,
                                sample_counts=train_acnt),
                            x=train_x,
                            dt=train_dt[train_matched],
                            u=train_u,
                            iterations=[(1500, 1), (500, 1e-1), (200, 1e-2),
                                        (200, 1e-3), (200, 1e-4)],
                            decay_rate=0.95,
                            epsilon=1e-6)

                        new_Mu, new_S2, new_mu, new_s2, new_w = split(
                            w, (1, 1, len(init_mu), len(init_s2), len(init_w)))

                        print(status)
                        print(nll)
                        p = lrxr.lr_probability(new_w, test_ux)
                        predictions = make_predictions(test_hy, p, devData)

                        pt = lrxr.lr_probability(new_w, train_ux)
                        train_predictions = make_predictions(train_hy, pt, trainData)

                        weights = []
                        weights.append(
                            ('MuSigma',
                             '%.2f+-%.2f' % (
                                np.squeeze(new_Mu),
                                np.sqrt(np.abs(np.squeeze(new_S2))))))

                        for i in range(len(train_arg_voc)):
                            argt = train_arg_voc.words[i]

                            mu_sigma0 = '%.2f+-%.2f' % \
                                (init_mu[i],
                                 np.sqrt(np.abs(np.squeeze(init_s2[i]))))
                            mu_sigma = '%.2f+-%.2f' % \
                                (new_mu[i],
                                 np.sqrt(np.abs(np.squeeze(new_s2[i]))))

                            weights.append(('mu_sigma',) + argt +
                                           (mu_sigma0, mu_sigma))

                        weights += \
                            sorted(zip(range(len(vocabulary)),
                                       vocabulary.words,
                                       new_w),
                                   key=lambda t: -t[-1])
                    elif mode == "model9":
                        parameters = {
                            'attr': mapping.name,
                            'mode': mode,
                            'bias_l2': bias_l2,
                            'feature_l2': feature_l2,
                            'xr': dict(xr_params._asdict()),
                            'mu': dict(mu_expmixture_params._asdict()),
                            'lam': dict(lam_params._asdict())
                        }

                        (w, nll, status) = lrxr.train_adadelta(
                            model9_obj, model9_grad,
                            w0=np.concatenate((
                                [init_pi, init_lp, init_ln],
                                init_mu, init_lam, init_w)),
                            wmin=np.concatenate((
                                [0, 1/1000.0, 1/1000.0],
                                np.ones_like(init_mu) * (-np.inf),
                                np.ones_like(init_lam) * (1/10.0),
                                np.ones_like(init_w) * (-np.inf))),
                            wmax=np.concatenate((
                                [1, 10.0, 10.0],
                                np.ones_like(init_mu) * np.inf,
                                np.ones_like(init_lam) * (10.0),
                                np.ones_like(init_w) * np.inf)),
                            parameters=Model9Parameters(
                                tau=tau0,
                                mu=mu_expmixture_params,
                                lam=lam_params,
                                l2=l2, t=1.0, xr=xr_params,
                                sample_counts=train_acnt),
                            x=train_x,
                            dt=train_dt[train_matched],
                            u=train_u,
                            iterations=[(1500, 1), (500, 1e-1), (200, 1e-2),
                                        (200, 1e-3), (200, 1e-4)],
                            decay_rate=0.95,
                            epsilon=1e-6)

                        new_pi, new_lp, new_ln, new_mu, new_lam, new_w = split(
                            w, (1, 1, 1, len(train_acnt), len(train_acnt),
                                len(init_w0)))

                        print "new_pi     : ", new_pi
                        print "1 / new_lp : ", 1 / new_lp
                        print "1 / new_ln : ", 1 / new_ln
                        print "1 / new_lam: ", 1 / np.mean(new_lam)
                        print "new_mu     : ", np.mean(new_mu)

                        print(status)
                        print(nll)
                        p = lrxr.lr_probability(new_w, test_ux)
                        predictions = make_predictions(test_hy, p, devData)

                        pt = lrxr.lr_probability(new_w, train_ux)
                        train_predictions = make_predictions(train_hy, pt, trainData)

                        weights = []
                        weights.append(('pi', new_pi))
                        weights.append(('lp', new_lp))
                        weights.append(('ln', new_ln))

                        print len(train_arg_voc)
                        print len(test_arg_voc)

                        for i in range(len(train_arg_voc)):
                            argt = train_arg_voc.words[i]

                            mu_sigma0 = '%.2f+-%.2f' % \
                                (init_mu[i], np.sqrt(np.abs(init_s2[i])))
                            mu_sigma = '%.2f+-%.2f' % \
                                (new_mu[i], 1/new_lam[i])

                            weights.append(('mu_sigma',) + argt +
                                           (mu_sigma0, mu_sigma))

                        weights += \
                            sorted(zip(range(len(vocabulary)),
                                       vocabulary.words,
                                       new_w),
                                   key=lambda t: -t[-1])
                    elif mode == "model11":
                        parameters = {
                            'attr': mapping.name,
                            'mode': mode,
                            'bias_l2': bias_l2,
                            'feature_l2': feature_l2,
                            'xr': dict(xr_params._asdict()),
                            'mu': dict(mu_expmixture_params._asdict()),
                            'lam': dict(lam_params._asdict())
                        }

                        w0 = np.concatenate((
                            [init_pi, init_lp, init_ln],
                            init_mu, init_delta, init_w))
                        wmin = np.concatenate((
                            [0, 0, 0],
                            np.ones_like(init_mu) * (-np.inf),
                            np.zeros_like(init_delta),
                            np.ones_like(init_w) * (-np.inf)))
                        wmax = np.concatenate((
                            [1, np.inf, np.inf],
                            np.ones_like(init_mu) * np.inf,
                            np.ones_like(init_delta) * np.inf,
                            np.ones_like(init_w) * np.inf))

                        w = w0
                        for k in [1, 1.5, 2.0, 2.5]:
                            (w, nll, status) = lrxr.train_adadelta(
                                model11_obj, model11_grad,
                                w0=w, wmin=wmin, wmax=wmax,
                                parameters=Model11Parameters(
                                    tau=tau0, l1=0.0, l2=l2,
                                    k=k, xr=xr_params,
                                    sample_counts=train_acnt),
                                x=train_x, dt=train_dt[train_matched],
                                u=train_u,
                                iterations=[(500, 1)],
                                decay_rate=0.95,
                                epsilon=1e-6)

                            new_pi, new_lp, new_ln, new_mu, new_delta, new_w = split(
                                w, (1, 1, 1, len(train_acnt), len(train_acnt),
                                    len(init_w0)))
                            print "new_pi     : ", new_pi
                            print "1 / new_lp : ", 1 / new_lp
                            print "1 / new_ln : ", 1 / new_ln
                            print "new_delta  : ", np.mean(new_delta)
                            print "new_mu     : ", np.mean(new_mu)

                        (w, nll, status) = lrxr.train_adadelta(
                            model11_obj, model11_grad,
                            w0=w, wmin=wmin, wmax=wmax,
                            parameters=Model11Parameters(
                                tau=tau0, l1=0.0, l2=l2,
                                k=3, xr=xr_params,
                                sample_counts=train_acnt),
                            x=train_x, dt=train_dt[train_matched],
                            u=train_u,
                            iterations=[(1500, 1), (500, 1e-1), (200, 1e-2),
                                        (200, 1e-3), (200, 1e-4)],
                            decay_rate=0.95,
                            epsilon=1e-6)

                        new_pi, new_lp, new_ln, new_mu, new_delta, new_w = split(
                            w, (1, 1, 1, len(train_acnt), len(train_acnt),
                                len(init_w0)))

                        print "new_pi     : ", new_pi
                        print "1 / new_lp : ", 1 / new_lp
                        print "1 / new_ln : ", 1 / new_ln
                        print "new_delta  : ", np.mean(new_delta)
                        print "new_mu     : ", np.mean(new_mu)

                        print(status)
                        print(nll)
                        p = lrxr.lr_probability(new_w, test_ux)
                        predictions = make_predictions(test_hy, p, devData)

                        pt = lrxr.lr_probability(new_w, train_ux)
                        train_predictions = make_predictions(train_hy, pt, trainData)

                        weights = []
                        weights.append(('pi', new_pi))
                        weights.append(('lp', new_lp))
                        weights.append(('ln', new_ln))

                        print len(train_arg_voc)
                        print len(test_arg_voc)

                        for i in range(len(train_arg_voc)):
                            argt = train_arg_voc.words[i]

                            mu_sigma0 = '%.2f+-%.2f' % \
                                (init_mu[i], init_delta[i])
                            mu_sigma = '%.2f+-%.2f' % \
                                (new_mu[i], new_delta[i])

                            weights.append(('mu_sigma',) + argt +
                                           (mu_sigma0, mu_sigma))

                        weights += \
                            sorted(zip(range(len(vocabulary)),
                                       vocabulary.words,
                                       new_w),
                                   key=lambda t: -t[-1])
                    else:
                        assert False

                    parameters_hash = sha1_structure(parameters)
                    subdir = "%s_%s" % (mapping.name, parameters_hash)
                    mkdir_p(os.path.join(output_dir, subdir))
                    subdir = os.path.join(output_dir, subdir)
                    print subdir

                    predictions.sort(key=lambda x: -x.p)
                    print "!! ", any(s.sample.id == "102053678478925824" for s in predictions)
                    train_predictions.sort(key=lambda x: -x.p)
                    heval_sids = set(r.sid for r in heval_results)
                    heval_r_by_sid = {r.sid: r for r in heval_results}
                    heval_y_by_attrsid = {(r.attr, r.sid): r.y for r in heval_results}

                    def compute_heval_y(s):
                        full_id = (mapping.name, s.id)
                        if full_id in heval_y_by_attrsid:
                            return 1 if heval_y_by_attrsid[full_id] > 0.5 \
                                else -1
                        else:
                            # FIXME!
                            if heval_r_by_sid[s.id].attr != mapping.name:
                                if heval_r_by_sid[s.id].y > 0.5:
                                    return -1
                                else:
                                    return 0
                            else:
                                return 0

                    heval_predictions = \
                        [x._replace(y=compute_heval_y(x.sample))
                         for x in predictions
                         if x.sample.id in heval_sids]
                    heval_predictions = [p for p in heval_predictions
                                         if p.y != 0]
                    N = sum(1 for x in predictions if x.y == 1)

                    PR = list(gen_tprf(predictions, N,
                                       sort=True, shuffle=True))

                    with open(os.path.join(subdir, 'tprf.csv'), 'w+') as f:
                        for T, P, R, F in PR:
                            f.write("%s\t%s\t%s\t%s\n" % (T, P, R, F))

                    train_max_tprf = fx.max_by(
                        gen_tprf(train_predictions, sort=True, shuffle=True),
                        key=lambda t: t[-1])

                    heval_max_tprf = fx.max_by(
                        gen_tprf(heval_predictions, sort=True, shuffle=True),
                        key=lambda t: t[-1])

                    test_max_tprf = fx.max_by(
                        gen_tprf(predictions, sort=True, shuffle=True),
                        key=lambda t: t[-1])

                    done_time = easytime.now()
                    time = done_time - last_time
                    last_time = done_time

                    print("%s/%s/%s in %s s" %
                          (mapping.name, mode, parameters, time))
                    print('%.2f\t%.2f\t%.2f' %
                          (train_max_tprf[-1],
                           test_max_tprf[-1],
                           heval_max_tprf[-1]))

                    with open(os.path.join(subdir, 'params.json'), 'w+') as f:
                        f.write(json.dumps(parameters) + "\n")

                    with open(os.path.join(subdir, 'scores.json'), 'w+') as f:
                        scores = {
                            'train': dict(train_max_tprf._asdict()),
                            'test': dict(test_max_tprf._asdict()),
                            'mturk': dict(heval_max_tprf._asdict()),
                            #'nll': nll,
                            #'status': status
                        }
                        print scores
                        f.write(json.dumps(scores) + "\n")

                    with open(os.path.join(subdir, 'test_predictions.json'), 'w+') as f:
                        for p in predictions:
                            f.write(json.dumps(p) + "\n")

                    with open(os.path.join(subdir, 'train_predictions.json'), 'w+') as f:
                        for p in train_predictions:
                            f.write(json.dumps(p) + "\n")

                    with open(os.path.join(subdir, 'weights.csv'), 'w+') as f:
                        for t in weights:
                            f.write('\t'.join(map(str, t)) + '\n')


def main(args):


    run_models(
        gigaword=args.gigaword,
        base_dir=args.datadir,
        unmatched_file_name=args.unlabeled,
        output_dir=args.outputdir,
        event_mapping=event_mapping,
        train_range=(args.trainstart, args.trainend),
        test_range=(args.teststart, args.testend),
        min_tweets_per_entity_pair=5,
        min_entity_pairs_per_feature=5,
        modes=[
            "baseline",
            "ad-hoc",
        ],
        feature_window_days=[1],
        ha_pos_range=[-10, 3],
        ha_neg_range=[-50, 3],
        heval_results=heval_results,
        annotated_ids=annotated_ids,
        well_aligned=well_aligned)

if __name__ == "__main__":
    # PROFILE = False

    # pr = None
    # if PROFILE:
    #     pr = cProfile.Profile()
    #     pr.enable()

    def ymd_timestamp(string):
        y, m, d = string.split('-')
        return easytime.ts(
            year=int(y), month=int(m), day=int(d))

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--datadir', required=True)
    parser.add_argument('-o', '--outputdir', required=True)
    parser.add_argument('-u', '--unlabeled', default=None, required=True)
    parser.add_argument('-g', '--gigaword', action='store_true', default=False)
    parser.add_argument(
        '-s', '--trainstart', type=ymd_timestamp, required=True)
    parser.add_argument(
        '-e', '--trainend', type=ymd_timestamp, required=True)
    parser.add_argument(
        '-b', '--teststart', type=ymd_timestamp, required=True)
    parser.add_argument(
        '-f', '--testend', type=ymd_timestamp, required=True)
    args = parser.parse_args()

    main(args)
