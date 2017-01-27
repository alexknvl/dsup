import json
import sys
import os

scores = []

for d in os.listdir(sys.argv[1]):
    scoreFile = "%s/%s/scores.json" % (sys.argv[1],d)
    if os.path.isfile(scoreFile):
        score = json.load(open(scoreFile))
        score['dir'] = d
        scores.append(score)

scores.sort(key=lambda a: a['mturk']['f1'], reverse=True)

for s in scores:
    print " ".join([str(s[x]) for x in ['dir', 'mturk']])
