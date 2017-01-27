import os, sys, os.path
import numpy as np
import ujson as json

def groupby(iterable, key):
  result = {}
  iterable = iter(iterable)
  for v in iterable:
    k = key(v)
    if k not in result:
      result[k] = []
    result[k].append(v)
  return result

def itemids(lst):
  result = {}
  for i, v in enumerate(lst):
    assert v not in result
    result[v] = i
  return result

def maxby(iterable, key):
  result = None
  result_key = None
  iterable = iter(iterable)
  for v in iterable:
    k = key(v)
    if result_key == None or result_key < k:
      result_key = k
      result = v
  return result

def read_predictions(input_dir):
  with open(os.path.join(input_dir, 'predOut')) as input_file:
    lines = (json.loads(line) for line in input_file)
    for line in lines:
      yield line

input_dir = sys.argv[1]
print list(read_predictions(input_dir))[0]
