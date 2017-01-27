#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import division

import sys
import os
import gzip

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

from FeatureExtractor import generate_binary_features

from common import Text, Sample, Edit, gen_tprf, mkdir_p, sha1_structure, normalize_attr
from vocab import Vocabulary
import data
import twitter as tw
import gigaword as gw

from collections import namedtuple
from recordtype import recordtype
from tabulate import tabulate

import result_store


def tsv_print(file, *args):
    file.write('\t'.join(str(s) for s in args) + '\n')


def read_unlabeled(gigaword, path, range):
    tweets = []
    ep_tweets = {}
    samples = gw.read_unmatched(path) if gigaword else tw.read_unmatched(path)
    neg_count = 0
    for s in samples:
        neg_count += 1
        if neg_count % 100000 == 0:
            print("read_preprocessed_negative: %s" % neg_count)

        if range[0] <= s.timestamp <= range[1]:
            ep_tweets.setdefault(s.args, []).append(s)
            tweets.append(s)

    return tweets, ep_tweets

Prediction = namedtuple('Prediction', ['y', 'p', 'sample'])
WindowSample = namedtuple('WindowSample', ['features', 'sample'])


def generate_data(tweets, feature_window_days, ep_tweets):
    result = []

    for s in tweets:
        samples = ep_tweets.get(s.args, [])

        dt = easytime.dt(days=feature_window_days)
        predicate = lambda x: x.timestamp <= s.timestamp and \
                              x.timestamp > (s.timestamp - dt)
        features = (x for x in samples if predicate(x))
        features = (f for x in features for f in x.features)
        features = sorted(set(features))
        result.append(WindowSample(features, s))

    return sorted(result, key=lambda r: r.sample.args)


def make_sparse_matrix(rows, column_count):
    x = scipy.sparse.lil_matrix((len(rows), column_count))
    for i, js in enumerate(rows):
        for j in js:
            x[i, j] = 1
    return x.tocsr()


def make_dense_vector(items, n):
    x = np.zeros(n)
    for k, v in items:
        x[k] = v
    return x


ModelData = recordtype('ModelData', ['weights', 'output_file'])


def main(sample_file, range, dataset_name, model_dir):
    dirs = [(path, attr, hash)
            for path, (attr, hash) in
            result_store.v3_get_all_output_dirs(model_dir)]

    vocabulary = Vocabulary(['__BIAS__'])

    def is_int(x):
        try:
            int(x)
            return True
        except:
            return False

    def read_weights(path):
        with open(path, 'r') as f:
            lines = (line.rstrip().split('\t') for line in f)
            lines = (line for line in lines if len(line) == 3 and is_int(line[0]))
            lines = ((line[1], float(line[2])) for line in lines)
            for line in lines:
                yield line

    models = {}
    for (path, attr, hash) in dirs:
        output_file = gzip.open(os.path.join(path, 'full_predictions.csv.gz'), 'w+')
        weights = read_weights(os.path.join(path, 'weights.csv'))
        weights = ((vocabulary.update(k), v) for k, v in weights)
        weights = dict(weights)

        print path

        models[path] = ModelData(
            weights=weights,
            output_file=output_file)
    print("n features: %s" % len(vocabulary))

    # for path, data in models.items():
    #     data.weights = make_dense_vector(data.weights, len(vocabulary))

    tweets, ep_tweets = read_unlabeled(
        gigaword=False, path=sample_file, range=range)

    print("Preprocessing/ExtractingFeatures is done.")
    for s in tweets:
        features = generate_binary_features(
            s.args, s.text.words, s.text.pos, s.text.ner)
        features = fx.concat(['__BIAS__'], features)
        features = (vocabulary.get(f) for f in features)
        features = sorted(set(f for f in features if f != -1))
        s.features = features
    print("Preprocessing/ExtractingFeatures is done.")

    print("Preprocessing/UnlabeledWindows started.")
    data_set = generate_data(
        tweets=tweets, feature_window_days=1, ep_tweets=ep_tweets)
    print("Preprocessing/UnlabeledWindows is done.")

    count = 0
    for s in data_set:
        count += 1
        if count % 10000 == 0:
            print count

        for path, model in models.iteritems():
            weights = model.weights
            k = sum(weights.get(f, 0) for f in s.features)
            p = 1 / (1 + np.exp(-k))
            model.output_file.write(str(p) + '\t' + json.dumps(s.sample) + "\n")

    for model in models.itervalues():
        model.output_file.close()


if __name__ == "__main__":
    np.seterr(all='raise', under='ignore')

    def ymd_timestamp(string):
        y, m, d = string.split('-')
        return easytime.ts(year=int(y), month=int(m), day=int(d))

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--samples', required=True)
    parser.add_argument('-d', '--models', required=True)
    parser.add_argument('-s', '--start', type=ymd_timestamp, required=True)
    parser.add_argument('-e', '--end', type=ymd_timestamp, required=True)
    parser.add_argument('-n', '--name', required=True)
    args = parser.parse_args()

    main(sample_file=args.samples,
         range=(args.start, args.end),
         dataset_name=args.name,
         model_dir=args.models)
