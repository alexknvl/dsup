import sys
import json
from datetime import *

def ParseDate(string):
    result = None
    try:
        result = datetime.strptime(string, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        result = datetime.strptime(string, '%a %b %d %H:%M:%S +0000 %Y')
    return result

for line in sys.stdin:
    js = json.loads(line.strip())
    #print js['wpEdits'][0]['timestamp']
    editDates = [datetime.fromtimestamp(json.loads(x)['timestamp']) for x in js['wpEdits']]
    tweetDate = ParseDate(js['tweet']['created_at'])
    editDiffs = [str(x - tweetDate) for x in editDates]
    editDatesStr = [x.strftime('%Y-%m-%d %H:%M') for x in editDates]
    print str(js['pred']) + "\t" +  str((js['y'], js['arg1'], js['arg2'], js['tweetDate'], editDatesStr, editDiffs, js['tweet']['words'], js['wpEdits']))
    
#    tweetDates = [ParseDate(x['created_at']) for x in js['tweet']]
#    tweetDates.sort()
#    firstTweetDate = tweetDates[0]
#    lastTweetDate = tweetDates[-1]
    
#    if len(js['wpEdits']) > 0:
        #print str(js['pred']) + "\t" +  str((js['y'], js['arg1'], js['arg2'], js['tweetDate'], firstTweetDate.strftime('%Y-%m-%d %H:%M'), lastTweetDate.strftime('%Y-%m-%d %H:%M'), len(js['tweet']), datetime.fromtimestamp(json.loads(js['wpEdits'][0])['timestamp']).strftime('%Y-%m-%d %H:%M')))
        #print ">>>>" + "\n>>>>>".join([x['words'] for x in js['tweet']])
        
