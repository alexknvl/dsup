import sys
import json

for line in sys.stdin:
    js = json.loads(line.strip())
    print js['y']
