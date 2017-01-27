#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys
import os

import numpy as np
import ujson as json

import fx
import easytime
import lrxr

import result_store as results
from common import Text, Sample, Edit, normalize_attr
from vocab import Vocabulary

from collections import namedtuple
from tabulate import tabulate


def meanvar(x, w):
    x = np.asarray(x)
    w = np.asarray(w)
    w = w / w.sum()
    mu = (x * w).sum()
    s2 = (x ** 2 * w).sum() - mu**2
    return mu, s2


YMD = namedtuple('YMD', 'year month day')


def ymd_to_timestamp_range(ymd):
    if ymd.month is None:
        return (easytime.ts(year=ymd.year, month=1, day=1),
                easytime.ts(year=ymd.year + 1, month=1, day=1))
    elif ymd.day is None:
        if ymd.month == 12:
            return (easytime.ts(year=ymd.year, month=12, day=1),
                    easytime.ts(year=ymd.year + 1, month=1, day=1))
        else:
            return (easytime.ts(year=ymd.year, month=ymd.month, day=1),
                    easytime.ts(year=ymd.year, month=ymd.month + 1, day=1))
    else:
        low = easytime.ts(year=ymd.year, month=ymd.month, day=ymd.day)
        return (low, low + easytime.dt(days=1))


def range_intersect(r1, r2):
    # Check that we are given well-formed ranges.
    assert r1[0] <= r1[1]
    assert r2[0] <= r2[1]

    # Make sure r1 starts earlier than r2.
    if r2[0] < r1[0]:
        return range_intersect(r2, r1)

    if r1[1] < r2[0]:
        # r1 ends before r2 starts.
        return None
    elif r2[0] <= r1[1] <= r2[1]:
        # r1 ends inside r2.
        return (r2[0], r1[1])
    elif r1[1] > r2[1]:
        # r1 ends after r2 ends.
        return r2
    else:
        assert False


def range_check(r, t):
    # Check that we are given a well-formed range.
    assert r[0] <= r[1]

    return r[0] <= t <= r[1]


def range_distance(r, t):
    # Check that we are given a well-formed range.
    assert r[0] <= r[1]

    if t <= r[0]:
        return t - r[0]
    elif r[0] <= t <= r[1]:
        return 0
    elif t >= r[1]:
        return t - r[1]


HEvalTime = namedtuple('HEvalTime', 'args dates')
Event = namedtuple('Event', 'source predicate args range')


def read_heval_times(path):
    result = {}

    def try_parse_date_list(s):
        dates = s.split(',')
        for i, d in enumerate(dates):
            try:
                parts = map(int, d.split('-'))
                if len(parts) == 0 or len(parts) > 3:
                    return None
                y = parts[0]
                m = None if len(parts) == 1 else parts[1]
                d = None if len(parts) <= 2 else parts[2]
                dates[i] = YMD(y, m, d)
            except:
                return None
        return dates

    with open(path) as f:
        lines = (line.rstrip().split('\t') for line in f)
        for line in lines:
            attr = normalize_attr(line[0])
            arg1 = line[1]
            arg2 = line[2]
            if len(line) <= 3:
                continue
            dates = try_parse_date_list(line[3])
            if dates is None:
                print("Couldn't parse dates for %s(%s, %s): %s" %
                      (attr, arg1, arg2, line[3]))
                continue
            result.setdefault(attr, {})[(arg1, arg2)] = dates

    return result


def read_wikidata_events(path):
    result = {}

    with open(path) as f:
        lines = (line.rstrip().split('\t') for line in f)
        for line in lines:
            #print(line)
            attr, arg1, arg2, date_str, prec, narg1, narg2, t1, t2 = line
            t1 = int(t1)
            t2 = int(t2)
            dates = result.setdefault(attr, {}) \
                .setdefault((narg1, narg2), set())
            dates.add((t1, t2))

    return result


HEVAL_ANNOTATED_TIMES_PATH = \
    '/home/konovalo/dsup_event/new-data/hand-annotated-events.csv'

WIKIDATA_ANNOTATED_TIMES_PATH = \
    '/home/konovalo/dsup_event/new-data/wikidata/all.csv'


def main(dir, train_range):
    heval_times = read_heval_times(HEVAL_ANNOTATED_TIMES_PATH)
    wikidata_events = read_wikidata_events(WIKIDATA_ANNOTATED_TIMES_PATH)

    print("Total hand-annotated attributes: %d" % len(heval_times))
    for k in heval_times.keys():
        print("  %s: %d" % (k, len(heval_times[k])))

    print("Total wikidata-annotated attributes: %d" % len(wikidata_events))
    for k in wikidata_events.keys():
        print("  %s: %d" % (k, len(wikidata_events[k])))

    events = {}

    for predicate, byarg in heval_times.items():
        for args, dates in byarg.items():
            term = (predicate,) + args
            ymds = (x for x in dates if x.year >= 1970)
            ranges = (ymd_to_timestamp_range(x) for x in ymds)
            ranges = \
                [Event(
                    source='hand-annotated',
                    predicate=predicate,
                    args=args,
                    range=r)
                 for r in ranges
                 if range_intersect(train_range, r) is not None]

            if len(ranges) > 0:
                events.setdefault(term, []).extend(ranges)

    for predicate, byarg in wikidata_events.items():
        for args, ranges in byarg.items():
            term = (predicate,) + args
            ranges = \
                [Event(
                    source='wikidata',
                    predicate=predicate,
                    args=args,
                    range=r)
                 for r in ranges
                 if range_intersect(train_range, r) is not None]

            if len(ranges) > 0:
                events.setdefault(term, []).extend(ranges)

    ha_attrs = set(e.predicate for es in events.values() for e in es
                   if e.source == 'hand-annotated')
    wd_attrs = set(e.predicate for es in events.values() for e in es
                   if e.source == 'wikidata')
    print("Matched hand-annotated attributes: %d" % len(ha_attrs))
    for a in ha_attrs:
        print("  %s: %d" %
              (a, len(set(e.args for es in events.values()
                          for e in es
                          if e.predicate == a
                          and e.source == 'hand-annotated'))))

    print("Matched hand-annotated attributes: %d" % len(wd_attrs))
    for a in wd_attrs:
        print("  %s: %d" %
              (a, len(set(e.args for es in events.values()
                          for e in es
                          if e.predicate == a
                          and e.source == 'wikidata'))))

    dirs = [(path, attr, hash)
            for path, (attr, mode1, mode2, hash) in
            results.v2_get_all_output_dirs(dir)]
    dirs += [(path, attr, hash)
             for path, (attr, hash) in
             results.v3_get_all_output_dirs(dir)]

    SYNONYMS = {
        'currentclub,currentteam,team': 'CurrentTeam',
        'death_place': 'DeathPlace',
        'spouse': 'Spouse'
    }

    sources = set(e.source
                  for es in events.values()
                  for e in es)

    for (path, attr, hash) in dirs:
        print("Starting evaluation of %s" % path)

        params = results.v2_read_params(path)
        weights = results.v2_read_weights(path)
        attr = normalize_attr(params['attr'])
        attr = SYNONYMS.get(attr, attr)

        predictions = {}
        matches = {}
        for p in results.v2_read_predictions(path, 'train'):
            term = (attr,) + p.s.args
            # if attr == 'DeathPlace' and p.s.args[0] == 'elizabeth taylor':
            #     print(term)
            #     print(term in events)
            #     if term in events:
            #         print(events[term])
            inrange = range_check(train_range, p.s.timestamp)
            if term in events and inrange:
                for e in events[term]:
                    matches.setdefault(e.source, set()).add(p.s.args)
                matches.setdefault('all', set()).add(p.s.args)
                predictions.setdefault(term, []).append(p)

        print("Matched *             : %d" %
              len(matches.get('all', set())))
        print("Matched hand-annotated: %d" %
              len(matches.get('hand-annotated', set())))
        print("Matched wikidata      : %d" %
              len(matches.get('wikidata', set())))

        offsets = {}

        for term in predictions.keys():
            attr, arg1, arg2 = term

            ranges = {s: [e.range for e in events[term] if e.source == s]
                      for s in sources}
            ranges['all'] = [e.range for e in events[term]]

            st = np.array([p.s.timestamp for p in predictions[term]])
            et = p.s.edits[0].timestamp if len(p.s.edits) > 0 else np.nan
            p = np.array([p.p for p in predictions[term]])

            mode = st[np.argmax(p)]
            mu, s2 = meanvar(st, p)

            def time_offset(ranges, time):
                distances = np.array([range_distance(r, time) for r in ranges])
                i = np.argmin(np.abs(distances))
                return distances[i] / easytime.dt(days=1)

            def add_offset(name, time):
                for s, rs in ranges.items():
                    if len(rs) > 0:
                        offsets.setdefault(s, {})\
                            .setdefault(name, [])\
                            .append(time_offset(rs, time))

            if not np.isnan(et) and range_check(train_range, et):
                add_offset('edit', et)
            add_offset('mode', mode)
            add_offset('mean', mu)

            if weights.ta_params is not None and not np.isnan(et):
                mu0, sigma0, mu, sigma = weights.ta_params.pairs[(arg1, arg2)]
                add_offset('ta0', et - mu0 * easytime.dt(days=1))
                add_offset('ta', et - mu * easytime.dt(days=1))

        for s in offsets:
            for m in offsets[s]:
                offsets[s][m] = (np.mean(offsets[s][m]), np.var(offsets[s][m]))
            offsets[s]['matches'] = len(matches.get(s, set()))

        with open(os.path.join(path, 'offsets.json'), 'w+') as f:
            print(json.dumps(offsets))
            f.write(json.dumps(offsets) + '\n')


if __name__ == "__main__":
    main(
        sys.argv[1],
        train_range=(easytime.ts(year=2008, month=9, day=1),
                     easytime.ts(year=2011, month=6, day=1)))
