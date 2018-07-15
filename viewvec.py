from pychord.constants.qualities import QUALITY_DICT
import numpy as np

root = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'Eb',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'Ab',
    9: 'A',
    10: 'Bb',
    11: 'B',
}

def tochord(num):
    if num == 0:
        return '-'
    num -= 1
    rootnum = 0
    qsum = len(QUALITY_DICT)
    while 1:
        if num >= qsum:
            num -= qsum
            rootnum += 1
        else:
            break
    return root[rootnum] + list(QUALITY_DICT.keys())[num]

a = ''
for i in range(481):
    a += tochord(i) + '\n'
a = a[:-1]
with open('metadata.tsv', 'w') as f:
    file = f.write(a)
