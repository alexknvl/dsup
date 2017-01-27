#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import division

import ujson as json

from collections import namedtuple


MTurkDevSample = namedtuple('MTurkDevSample', 'sid attr args counts y')


def read_mturk_dev_file(path):
    with open(path) as f:
        lines = (json.loads(line) for line in f)
        for sid, attr, arg1, arg2, counts in lines:
            (yes, no, ns) = counts
            y = (yes + ns/2) / (yes + no + ns)
            yield MTurkDevSample(sid, attr, (arg1, arg2), counts, y)


EventMapping = namedtuple('EventMapping', 'name attributes')
EventMappingAttribute = namedtuple('EventMappingAttribute',
                                   'name infoboxes symmetric invert')


def read_event_mapping_file(path):
    def parse_attribute(name, j):
        symmetric = j.get('symmetric', False)
        invert = j.get('invert', False)
        if 'infoboxes' in j:
            infoboxes = set(j['infoboxes'])
        else:
            infoboxes = None

        return EventMappingAttribute(name, infoboxes, symmetric, invert)

    def parse_mapping(name, j):
        attributes = []
        for k, v in j['attributes'].items():
            attributes.append(parse_attribute(k, v))
        return EventMapping(name, attributes)

    result = []
    with open(path) as f:
        j = json.loads(f.read())
        for k, v in j.items():
            result.append(parse_mapping(k, v))
    return result
