#!/usr/bin/env python

import csv
import os
from pychord import Chord
from pychord.quality import Quality
from pychord.utils import NOTE_VAL_DICT
from pychord.parser import check_note
import re
from pychord.constants import QUALITY_DICT, NOTE_VAL_DICT, SCALE_VAL_DICT
import copy
import unicodedata
import tqdm
import numpy as np


class Quality(Quality):
    TENSIONDICT = {
        '6': 9,
        'b9': 13,
        '-9': 13,
        '9':  14,
        '#9': 15,
        '+9': 15,
        '11': 17,
        '#11': 18,
        '+11': 18,
        'b13': 20,
        '-13': 20,
        '13': 21,
        '#13': 22,
        '+13': 22,
        'add11': 17,
        'add9': 14,
        # 'add12': 12
    }

    def __init__(self, quality, tensions):
        if quality not in QUALITY_DICT:
            raise ValueError("unknown quality {}".format(quality))
        self._quality = quality
        self._tension = tensions
        self.components =copy.deepcopy(QUALITY_DICT[quality])
        if tensions:
            self.components.extend([self.TENSIONDICT[tension] for tension in tensions])


class Chord(Chord):
    dairi = {
        '': '△',
        'min': 'm',
        '7+5': 'aug7',
        'dim6': 'dim7',
        '2': 'add9',
        '4': 'add11',
        '6/9': '69',
        'augM7': 'M7+5',
        'm-5': 'dim',
        'omit5': '5',
        'm7b5': 'm7-5',
        '7add9': '9',
        'maj7': 'M7',
        '-5': 'dim',
        'maj9': 'M9',
        '8': '△'
    }

    def _parse(self, chord):
        root, quality, appended, on = self.parse(chord)
        self._root = root
        self._quality = quality
        self._appended = appended
        self._on = on

    def parse(self, chord):
        if len(chord) > 2 and chord[1:3] in ("bb", "##"):
            root = chord[:3]
            rest = chord[3:]
        elif len(chord) > 1 and chord[1] in ("b", "#"):
            root = chord[:2]
            rest = chord[2:]
        else:
            root = chord[:1]
            rest = chord[1:]
        check_note(root, chord)
        on_chord_idx = rest.find("/")
        if on_chord_idx >= 0:
            on = rest[on_chord_idx + 1:]
            rest = rest[:on_chord_idx]
            check_note(on, chord)
        else:
            on = None
        rest = unicodedata.normalize('NFKC', rest)
        if re.search(r'\((.+)\)', rest):
            tensions = [i for i in re.split('[,(\)\()]', re.search(r'\((.+)\)', rest).group(1)) if i != '']
            rest = rest[:rest.find('(')]

            if tensions == ['b5'] or tensions == ['-5']:
                tensions = []
                rest = rest + '-5'
            elif tensions == ['#5'] or tensions == ['+5']:
                tensions = []
                rest = rest + '+5'
            elif len(tensions) == 1 and any([i in tensions[0] for i in ['sus', 'M7', '単音', 'add', 'omit']]) and rest == '':
                rest = tensions[0]
                tensions = []
        else:
            tensions = None
        if rest in self.dairi:
            rest = self.dairi[rest]
        if rest in QUALITY_DICT:
            quality = Quality(rest, tensions)
        else:
            raise ValueError("Invalid chord {}: Unknown quality {}".format(chord, rest))
        # TODO: Implement parser for appended notes
        appended = []
        return root, quality, appended, on

    def components(self, visible=True):
        """ Return the component notes of chord

        :param bool visible: returns the name of notes if True else list of int
        :rtype: list[(str or int)]
        :return component notes of chord
        """
        if self._on:
            self._quality.append_on_chord(self.on, self.root)

        comp = self._quality.get_components(root=self._root, visible=visible)
        if any([i < 0 for i in comp]):
            comp = [i+12 for i in comp]
        return comp

def main():
    file = 'data/chord.csv'
    if os.path.isfile(file):
        with open(file, 'r') as f:
            reader = csv.reader(f)
            chords = list(reader)
    data = []
    pbar = tqdm.tqdm(total=len(chords)*12)
    for num, title, chord in chords:
        for k in range(0, 12):
            note = []
            for j in chord.split():
                if 'N,C' in j or 'N.C' in j or '>' in j or j[0] == '/' or j[:2] == '(/' or '↓' in j or j == '-' or j == '':
                    continue
                if j[0] == '(' and j[-1] == ')':
                    j = j[1:-1]
                elif not '(' in j and j[-1] == ')':
                    j = j[:-1]
                elif not ')' in j and j[0] == '(':
                    j = j[1:]
                try:
                    chord1 = Chord(j)
                    chord1.transpose(k)
                    notes = chord1.components(False)
                except Exception as e:
                    # print(e, 'error', j)
                    notes = []
                note.append(notes)
            data.append(note)
            pbar.update(1)
    pbar.close()
    return data


def to_shot(chord, window=2):
    # chord = main()
    if chord is None:
        with open('data/intchord.csv', 'r') as f:
            reader = csv.reader(f)
            chord = list(reader)
            chord = [[[int(k) for k in j[1:-1].split(',') if k != ''] for j in i] for i in chord]

    sum1 = sum([len(i) for i in chord])
    list0 = []
    shot = []
    shotmax = max([max([max([0] + j) for j in i]) for i in chord])

    def func(i, k):
        if k < 0 or k >= len(i):
            return [0]* shotmax
        return [1 if l in i[k] else 0 for l in range(shotmax)]

    for i in chord:
        for j in range(len(i)):
            shot.append([func(i, j + k) for k in range(-window, window+1)])
    shot = np.array(shot)
    return shot

def oneshot(chords=None):
    a = dict([(j, i) for i, j in list(enumerate(QUALITY_DICT.keys()))])
    file = 'data/ufret.csv'
    if chords is None:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            chords = list(reader)
    data = []
    pbar = tqdm.tqdm(total=len(chords) * 12)
    for num, title, hito, chord in chords:
        for k in range(0, 12):
            note = []
            for j in chord.split('|'):
                if '♭' in j:
                    j = j.replace('♭', 'b')
                if 'N.C' in j:
                    continue
                try:
                    chord1 = Chord(j)
                    chord1.transpose(k)
                    notes = 1 + len(QUALITY_DICT) * NOTE_VAL_DICT[chord1.root] + \
                            dict([(j,i) for i, j in list(enumerate(QUALITY_DICT.keys()))])[chord1.quality.quality]
                    # max: 1 + 40 * 11 + 40 = 481
                except Exception as e:
                    notes = 0
                note.append(notes)
            if len(note) < 10:
                pass
            if len(data) == 373:
                pass
            data.append(note)
            pbar.update(1)
    pbar.close()
    file = 'data/ufret_emb.csv'
    with open(file, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')  # 改行コード（\n）を指定しておく
        writer.writerows(data)  # 2次元配列も書き込める
    return data

def to_one_shot(chord, window=2):
    # chord = main()
    if chord is None:
        with open('data/one_shot.csv', 'r') as f:
            reader = csv.reader(f)
            # chord = list(reader)
            chord = [[int(j) for j in i] for i in chord]
            return chord

    shotmax = 625
    shot = []

    def func(i, k):
        if k < 0 or k >= len(i):
            return [0] * shotmax
        return [1 if l == i[k] else 0 for l in range(shotmax)]

    pbar = tqdm.tqdm(total=len(chord))
    for i in chord:
        for j in range(len(i)):
            shot.append([func(i, j + k) for k in range(-window, window+1)])
        pbar.update(1)
    pbar.close()
    shot = np.array(shot)

    file = 'data/multi_shot.csv'
    with open(file, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')  # 改行コード（\n）を指定しておく
        writer.writerows(shot)  # 2次元配列も書き込める
    return shot

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

def count():
    with open('data/one_shot.csv', 'r') as f:
        reader = csv.reader(f)
        shot = list(reader)
        shot = [[int(j) for j in i] for i in shot]

    c = dict()

    for i in shot:
        for j in i:
            j = tochord(j)
            c[j] = c.get(j, 0) + 1
    pass

# to_one_shot(None)


if __name__ == '__main__':
    # count()
    a = oneshot()