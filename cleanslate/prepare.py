#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import division

import sys
import os
import logging
import gzip
import cPickle as pickle

import numpy as np
import scipy.sparse
import itertools

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from FeatureExtractor import generate_binary_features

import fx
import easytime
from vocab import Vocabulary
from collections import namedtuple
import twitter as tw


def check(x, enabled=True):
    assert x


def setup_logging():
    logger = logging.getLogger('dsup_event.prepare')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('debug.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s/%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()


def read_unmatched(path, skip_ids, ranges):
    samples = []
    set_ids = [list() for _ in ranges]
    arg_ids = dict()

    logger.debug("Reading unmatched samples.")

    unmatched_count = 0
    for s in tw.read_unmatched(path):
        unmatched_count += 1
        if unmatched_count % 500000 == 0:
            logger.debug("Read %s unmatched samples." % unmatched_count)

        if s.id in skip_ids:
            continue

        in_range = [r[0] <= s.timestamp <= r[1] for r in ranges]
        if not any(in_range):
            continue

        i = len(samples)
        samples.append(s)
        arg_ids.setdefault(s.args, []).append(i)

        for j, b in enumerate(in_range):
            if b:
                set_ids[j].append(i)

    for s in set_ids:
        s.sort(key=lambda i: samples[i].timestamp)

    assert(sorted(fx.concat(*set_ids)) == range(len(samples)))
    assert(sorted(fx.concat(*arg_ids.values())) == range(len(samples)))

    logger.debug("Finished reading unmatched samples.")

    return samples, set_ids, arg_ids


def read_matched(base_path, attr_group, ranges):
    """
    Reads files with matched samples corresponding to the given
    attribute group. Sorts samples into multiple date ranges.
    """

    samples = []
    set_ids = [list() for _ in ranges]
    arg_ids = dict()
    labeled_ids = set()
    read_ids = set()

    logger.debug("Reading attribute group '%s'." % ','.join(attr_group))

    for attr in attr_group:
        logger.debug("Reading matched attribute '%s'." % attr)

        for s in tw.read_matched(os.path.join(base_path, attr)):
            in_range = [r[0] <= s.timestamp <= r[1] for r in ranges]
            if not any(in_range):
                continue

            full_id = (s.id, s.args)
            if full_id in read_ids:
                continue
            read_ids.add(full_id)

            i = len(samples)
            samples.append(s)

            arg_ids.setdefault(s.args, []).append(i)

            for j, b in enumerate(in_range):
                if b:
                    set_ids[j].append(i)

            # Keep track of the IDs in the labeled dataset so
            # we can exclude them from the unlabeled data.
            labeled_ids.add(s.id)

    for s in set_ids:
        s.sort(key=lambda i: samples[i].timestamp)

    assert(sorted(fx.concat(*set_ids)) == range(len(samples)))
    assert(sorted(fx.concat(*arg_ids.values())) == range(len(samples)))

    logger.debug("Finished reading attribute group '%s'." %
                 ','.join(attr_group))

    return samples, set_ids, arg_ids, labeled_ids


def read_data(base_dir, unmatched_file_name, attr_groups,
              train_range, test_range):
    attr_samples = {}
    attr_arg_ids = {}
    attr_train_ids = {}
    attr_dev_ids = {}
    labeled_ids = set()

    for attr_group in attr_groups:
        attr_group_name = ','.join(attr_group)

        samples, set_ids, arg_ids, new_labeled_ids = read_matched(
            base_path=base_dir,
            attr_group=attr_group,
            ranges=[train_range, test_range])

        attr_samples[attr_group_name] = samples
        attr_train_ids[attr_group_name] = set_ids[0]
        attr_dev_ids[attr_group_name] = set_ids[1]
        attr_arg_ids[attr_group_name] = arg_ids
        labeled_ids.update(new_labeled_ids)

    for attr_group in [['unmatched']]:
        attr_group_name = ','.join(attr_group)

        samples, set_ids, arg_ids = read_unmatched(
            path=os.path.join(base_dir, unmatched_file_name),
            skip_ids=labeled_ids,
            ranges=[train_range, test_range])

        attr_samples[attr_group_name] = samples
        attr_train_ids[attr_group_name] = set_ids[0]
        attr_dev_ids[attr_group_name] = set_ids[1]
        attr_arg_ids[attr_group_name] = arg_ids

    return attr_samples, attr_arg_ids, attr_train_ids, attr_dev_ids


def feature_windows(samples, ids, arg_ids, feature_matrix,
                    feature_window_days):
    result = dict()

    for i in ids:
        s = samples[i]
        dt = easytime.dt(days=feature_window_days)
        window = [feature_matrix[j]
                  for j in arg_ids.get(s.args, [])
                  if samples[j].timestamp <= s.timestamp
                  and samples[j].timestamp > (s.timestamp - dt)]
        window_features = sorted(set(fx.concat(*window)))
        result[i] = window_features

    return result


def run_count_argt(samples, ids):
    argt_cnt = []
    argt_voc = Vocabulary()
    matched_data = (i for i in ids if len(samples[i].edits) != 0)
    matched_data = fx.run_group_by(matched_data, lambda i: samples[i].args)
    for j, (argt, ixs) in enumerate(matched_data):
        assert j == argt_voc.update(argt)
        argt_cnt.append(len(list(ixs)))
    argt_cnt = np.array(argt_cnt, dtype=np.int)
    return argt_voc, argt_cnt


def make_sparse_matrix(rows, column_count):
    x = scipy.sparse.lil_matrix((len(rows), column_count))
    for i, js in enumerate(rows):
        for j in js:
            x[i, j] = 1
    return x.tocsr()


def filter_low_frequency_arg_tuples(
        ag_names,
        attr_samples, attr_arg_ids, attr_train_ids, attr_dev_ids,
        argt_counts, min_tweets_per_entity_pair):
    discard_args = set(args
                       for args, cnt in argt_counts.iteritems()
                       if cnt < min_tweets_per_entity_pair)

    for ag in ag_names:
        attr_samples[ag], remapping = \
            fx.remap(attr_samples[ag], lambda i, s: s.args not in discard_args)
        attr_train_ids[ag] = \
            [remapping[i] for i in attr_train_ids[ag] if remapping[i] != -1]
        attr_dev_ids[ag] = \
            [remapping[i] for i in attr_dev_ids[ag] if remapping[i] != -1]

        for k, v in attr_arg_ids[ag].iteritems():
            attr_arg_ids[ag][k] = \
                [remapping[i] for i in v if remapping[i] != -1]

        assert(sorted(fx.concat(attr_train_ids[ag], attr_dev_ids[ag])) ==
               range(len(attr_samples[ag])))
        assert(sorted(fx.concat(*attr_arg_ids[ag].values())) ==
               range(len(attr_samples[ag])))


def extract_features(
        ag_names, attr_samples, attr_train_ids, attr_dev_ids,
        no_features_from_unmatched_samples):
    vocabulary = Vocabulary(['__BIAS__'])
    attr_features = dict()

    for ag in ag_names:
        samples = attr_samples[ag]
        train_ids = attr_train_ids[ag]
        dev_ids = attr_dev_ids[ag]
        features = [None] * len(samples)

        def gen_features(sample, freeze_voc):
            fs = fx.concat(['__BIAS__'], generate_binary_features(
                sample.args, sample.text.words,
                sample.text.pos, sample.text.ner))
            return sorted(set(vocabulary.get(f) for f in fs)) if freeze_voc \
                else sorted(set(vocabulary.update(f) for f in fs))

        freeze_voc = ag == 'unmatched' and no_features_from_unmatched_samples
        for i in train_ids:
            features[i] = gen_features(samples[i], freeze_voc)
        for i in dev_ids:
            features[i] = gen_features(samples[i], True)
        attr_features[ag] = features

    return vocabulary, attr_features


def prepare_data(
        attr_groups,
        matched_tweets_dir,
        unmatched_file,
        train_range,
        test_range,
        min_tweets_per_entity_pair,
        min_entity_pairs_per_feature,
        window_size,
        no_features_from_unmatched_samples):
    ag_names = [','.join(ag) for ag in attr_groups + [['unmatched']]]
    logger.debug("Attribute groups: %s." % ';'.join(ag_names))

    logger.info("Step 1. Reading the data.")
    attr_samples, attr_arg_ids, attr_train_ids, attr_dev_ids = read_data(
        base_dir=matched_tweets_dir,
        unmatched_file_name=unmatched_file,
        attr_groups=attr_groups,
        train_range=train_range,
        test_range=test_range)
    for ag in ag_names:
        logger.debug(
            "%s: samples=%s, arg_tuples=%s, train_set=%s, dev_set=%s" %
            (ag, len(attr_samples[ag]), len(attr_arg_ids[ag]),
             len(attr_train_ids[ag]), len(attr_dev_ids[ag])))

    logger.info("Step 2. Sorting samples by argument tuples.")
    for ag in ag_names:
        samples = attr_samples[ag]
        train_ids = attr_train_ids[ag]
        dev_ids = attr_dev_ids[ag]
        train_ids.sort(key=lambda i: samples[i].args)
        dev_ids.sort(key=lambda i: samples[i].args)

    logger.info("Step 3. Computing argument tuple frequencies.")
    argt_counts = fx.bag(s.args for ag in ag_names
                         for s in attr_samples[ag])

    logger.info("Step 4. Filtering low frequency argument tuples.")
    filter_low_frequency_arg_tuples(
        ag_names,
        attr_samples, attr_arg_ids, attr_train_ids, attr_dev_ids,
        argt_counts, min_tweets_per_entity_pair)
    del argt_counts
    for ag in ag_names:
        logger.debug(
            "%s: samples=%s, arg_tuples=%s, train_set=%s, dev_set=%s" %
            (ag, len(attr_samples[ag]), len(attr_arg_ids[ag]),
             len(attr_train_ids[ag]), len(attr_dev_ids[ag])))

    logger.info("Step 5. Extracting features.")
    vocabulary, attr_features = extract_features(
        ag_names, attr_samples, attr_train_ids, attr_dev_ids,
        no_features_from_unmatched_samples)
    logger.debug("Total features after this step: %s." % len(vocabulary))

    logger.info("Step 6. Computing feature frequencies.")
    feature_eps = [None] * len(vocabulary)
    for i, _ in enumerate(feature_eps):
        feature_eps[i] = set()

    for ag in ag_names:
        samples = attr_samples[ag]
        train_ids = attr_train_ids[ag]
        features = attr_features[ag]
        for i in train_ids:
            s = samples[i]
            for f in features[i]:
                feature_eps[f].add(s.args)
    for i, _ in enumerate(feature_eps):
        feature_eps[i] = len(feature_eps[i])

    logger.info("Step 7. Filtering low frequency features.")
    remapping = vocabulary.compress(
        lambda i, w: feature_eps[i] >= min_entity_pairs_per_feature)

    for ag in ag_names:
        features = attr_features[ag]
        for i, fs in enumerate(features):
            features[i] = [remapping[j] for j in fs if remapping[j] != -1]
    del remapping
    feature_eps = None
    logger.debug("Total features after this step: %s." % len(vocabulary))

    logger.info("Step 8. Combining features into feature windows.")
    attr_feature_windows = dict()

    for ag in ag_names:
        samples = attr_samples[ag]
        features = attr_features[ag]
        arg_ids = attr_arg_ids[ag]
        feature_windows = [None] * len(samples)

        for i, s in enumerate(samples):
            dt = easytime.dt(days=window_size)
            window = [features[j]
                      for j in arg_ids.get(s.args, [])
                      if samples[j].timestamp <= s.timestamp
                      and samples[j].timestamp > (s.timestamp - dt)]
            window = sorted(set(fx.concat(*window)))
            feature_windows[i] = window
        attr_feature_windows[ag] = feature_windows

    logger.info("Step 9. Converting data into numpy matrices.")
    attr_matrices = {}
    for ag in ag_names:
        samples = attr_samples[ag]
        features = attr_feature_windows[ag]
        train_ids = attr_train_ids[ag]
        st = np.array([s.timestamp for s in samples])
        et = np.array([s.timestamp if len(s.edits) != 0 else np.nan
                       for s in samples])
        dt = (et - st) / easytime.dt(days=1)
        x = make_sparse_matrix(features, len(vocabulary))

        argt_voc, argt_cnt = run_count_argt(samples, train_ids)
        attr_matrices[ag] = (x, dt, argt_voc, argt_cnt)

    # logger.info("Step X. Saving data.")

    # def save_data(obj, *path):
    #     logger.debug("Saving %s." % os.path.join(output_dir, *path))
    #     with gzip.open(os.path.join(output_dir, *path), 'w+') as f:
    #         pickle.dump(obj, f)

    # save_data(attr_samples, 'attr_samples.dat.gz')
    # save_data(attr_train_ids, 'attr_train_ids.dat.gz')
    # save_data(attr_dev_ids, 'attr_dev_ids.dat.gz')
    # save_data(attr_arg_ids, 'attr_arg_ids.dat.gz')

    # save_data(vocabulary, 'vocabulary.dat.gz')
    # save_data(attr_features, 'attr_features.dat.gz')
    # save_data(attr_feature_windows, 'attr_feature_windows.dat.gz')

    return (attr_samples, attr_train_ids, attr_dev_ids, attr_arg_ids,
            vocabulary, attr_features, attr_feature_windows,
            attr_matrices)


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
    print(by_timediff.shape)
    print(is_unmatched.shape)
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

# Data(matched_tweets_dir, attribute_list, unmatched_file)
# Split(train_range, test_range)
# FeatureExtraction(min_tweets_per_entity_pair, min_entity_pairs_per_feature)
# Window(window_size)

# LR(XR) heuristic labelling mode
Baseline = namedtuple('Baseline', [])
HeuristicAlignment = namedtuple(
    'HeuristicAlignment',
    ['positive_window', 'negative_window'])
AutomaticAlignment = namedtuple(
    'AutomaticAlignment',
    ['mu0', 'sigma0'])
LRXRParams = namedtuple('LRXRParams', ['bias_l2', 'feature_l2', 'xr', 'pex'])

if __name__ == "__main__":
    attr_groups = [s.split(',') for s in sys.argv[1].split(';')]
    output_dir = sys.argv[2]
    matched_tweets_dir = "../training_data2"
    unmatched_file = "negative_all_50.gz"
    train_range = (easytime.ts(year=2008, month=9, day=1),
                   easytime.ts(year=2011, month=6, day=1))
    test_range = (easytime.ts(year=2011, month=6, day=5),
                  easytime.ts(year=2012, month=1, day=1))
    min_tweets_per_entity_pair = 5
    min_entity_pairs_per_feature = 5
    window_size = 1
    no_features_from_unmatched_samples = False
    alignment_modes = [
        Baseline(),
        HeuristicAlignment(
            positive_window=[-10, 3],
            negative_window=[-50, 3]),
        AutomaticAlignment(
            mu0=15.0,
            sigma0=30.0**2)]
    lrxr_params = LRXRParams(
        bias_l2=10.0,
        feature_l2=25.0,
        xr=0,
        pex=0.5)

    np.seterr(all='raise', under='warn')
    np.random.seed(13241)

    (attr_samples, attr_train_ids, attr_dev_ids, attr_arg_ids,
     feature_vocabulary, attr_features, attr_feature_windows,
     attr_matrices) = \
        prepare_data(
            attr_groups=attr_groups,
            matched_tweets_dir=matched_tweets_dir,
            unmatched_file=unmatched_file,
            train_range=train_range,
            test_range=test_range,
            min_tweets_per_entity_pair=min_tweets_per_entity_pair,
            min_entity_pairs_per_feature=min_entity_pairs_per_feature,
            window_size=window_size,
            no_features_from_unmatched_samples=
            no_features_from_unmatched_samples)

    initial_weights = np.random.rand(len(feature_vocabulary))
    initial_weights[0] = 0

    (ux, _, _, _) = attr_matrices['unmatched']

    for attr_group in attr_groups:
        ag = ','.join(attr_group)

        for mode in alignment_modes:
            samples = attr_samples[ag]
            (x, dt, argt_voc, argt_cnt) = attr_matrices[ag]

            if isinstance(mode, Baseline):
                y = np.ones_like(dt)
            elif isinstance(mode, HeuristicAlignment):
                y = np.squeeze(np_classify_by_timediff(
                    mode.positive_window, mode.negative_window, -dt))
            elif isinstance(mode, AutomaticAlignment):
                y = None
            else:
                assert False

            x = make_sparse_matrix([s for s in samples],
                                   len(feature_vocabulary))

            if xr == "lr":
                (w, nll, status) = lrxr.lr_train(
                    w=initial_weights,
                    x=train_ux[train_hy != 0],
                    y=train_hy[train_hy != 0],
                    l2=10.0,
                    tol=1e-6)
                w_lr = w
                print(status)
                print(nll)
                p = lrxr.lr_predict(w, test_ux)
                predictions = make_predictions(test_hy, p, devData)
                weights = sorted(zip(vocabulary.words, w),
                                 key=lambda t: -t[1])
            elif xr == "lrxr":
                nonzero = np.nonzero(train_hy[train_matched])
                (w, nll, status) = lrxr.lrxr_train(
                    w=initial_weights,
                    x=train_x[nonzero],
                    y=train_hy[nonzero * train_matched],
                    u=train_u,
                    l2=10.0,
                    t=1.0,
                    p_ex=p_ex,
                    xr=10.0 * train_u.shape[0],
                    tol=1e-2)
                print(status)
                print(nll)
                p = lrxr.lr_predict(w, test_ux)
                predictions = make_predictions(test_hy, p, devData)
                weights = sorted(zip(vocabulary.words, w),
                                 key=lambda t: -t[1])
            elif xr == "lrxrta":
                if w_lr is not None:
                    w = w_lr
                else:
                    (w, nll, status) = lrxr.lr_train(
                        w=initial_weights,          # : F
                        x=train_ux[train_hy != 0],  # : (...) ⊗ F
                        y=train_hy[train_hy != 0],  # : (...) ⊗ F
                        l2=10.0,
                        tol=1e-6)

                l2 = np.ones_like(initial_weights)
                l2 *= 25.0
                l2[0] *= 10

                MU_0 = 15.0
                S2_0 = 30.0**2

                MU = 10.0
                S2 = 15.0**2

                mu = np.ones(len(train_acnt)) * MU
                s2 = 5.0**2

                all_times = np.array(
                    [v[-1].timestamp / easytime.dt(days=1)
                     for v in trainData])
                tau0 = np.max(all_times) - np.min(all_times)

                print(mu.shape)
                print(w.shape)
                print(train_x.shape)
                print(train_dt[train_matched].shape)

                (w, nll, status) = lrxr.lrxrta_train(
                    wf=np.concatenate(([MU], [S2], [s2], mu, w)),
                    acnt=train_acnt,
                    tau0=tau0,
                    MU_0=MU_0,
                    S2_0=S2_0,
                    x=train_x,
                    dt=train_dt[train_matched],
                    u=train_u,
                    l2=l2,
                    t=1.0,
                    p_ex=p_ex,
                    xr=0,  # train_u.shape[0],
                    tol=1e-6)

                new_MU = w[0]
                new_S2 = w[1]
                new_s2 = w[2]
                new_mu = w[3:3+len(train_acnt)]
                new_w = w[3+len(train_acnt):]

                print(status)
                print(nll)
                p = lrxr.lr_predict(new_w, test_ux)
                predictions = make_predictions(test_hy, p, devData)

                weights = [('__TAU__', tau0),
                           ('__MU_0__', MU_0),
                           ('__S_0__', np.sqrt(S2_0)),
                           ('__MU__', new_MU),
                           ('__S__', np.sqrt(np.abs(new_S2))),
                           ('__s__', np.sqrt(np.abs(new_s2)))]

                weights += [('mu_' + '_'.join(train_arg_voc.words[i]),
                             new_mu[i])
                            for i in xrange(train_acnt)]

                weights += \
                    sorted(zip(vocabulary.words,
                               w[3 + len(train_acnt):]),
                           key=lambda t: -t[1])
