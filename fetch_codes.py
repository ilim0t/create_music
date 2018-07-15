#!/usr/bin/env python

import os
import time
import urllib.request
from bs4 import BeautifulSoup
import pickle
import re
import csv
import socks
import socket
import tqdm
from selenium import webdriver
import random


class ChordWiki(object):
    url = 'https://ja.chordwiki.org'

    def __init__(self, n):
        self.n = n

    def getlinks(self, a):
        if a == 1:
            url = self.url + '/ranking.html'
        else:
            url = self.url + '/ranking' + str(a) + '.html'
        html = urllib.request.urlopen(url)
        soup = BeautifulSoup(html, "html.parser")
        lists = soup.find_all("table", {"class": "ranking"})[0].findAll('tr')
        sets = [[int(i.find('td').text[:-1]), i.findAll('td')[4].contents[0].text, self.url + i.findAll('td')[4].contents[0].attrs['href']] for
                i in lists]  # set= [no, title, url]
        return sets

    def getchord(self, url):
        html = urllib.request.urlopen('https://ja.chordwiki.org/wiki.cgi?c=edit&t=' + url.split('/')[-1])
        soup = BeautifulSoup(html, "html.parser")
        data = soup.find('textarea', {'name': 'chord'})
        try:
            chord = [data.text.split('title')[1].split('}')[0], data.text]
        except:
            print('error')
            chord = [data.text.split('t')[1].split('}')[0], data.text]
        if chord[0][0] == ':':
            chord[0] = chord[0][1:]
        return chord

    def tochord(self, string):
        chords = [i.split(']')[0] for i in string.split('[') if ']' in i]
        # chords = [i for i in chords if i != '<' and i != 'N.C.']
        return ' '.join(chords)


class UFRET(object):
    url = 'http://www.ufret.jp/song.php?data='

    def __init__(self):
        self.driver = webdriver.Safari()

    def tochord(self, num):
        self.driver.get(self.url + str(num), )
        html = self.driver.page_source.encode('utf-8')
        soup = BeautifulSoup(html, "html.parser")
        chords = [i.text for i in soup.select_one('#blyodnijb').find_all('rt')]
        title = soup.find('div', {'class': 'well well-sm'}).h1.strong.text
        hito = soup.find('div', {'class': 'well well-sm'}).h1.span.text

        return title, hito[2:], '|'.join(chords)

    def main(self, interval=1):
        file = 'data/ufret.csv'
        nmax = 1000
        self.chords = []
        self.pbar = tqdm.tqdm(total=nmax)
        if os.path.isfile(file):
            with open(file, 'r') as f:
                reader = csv.reader(f)
                self.chords = list(reader)

        f = open(file, 'a')
        writer = csv.writer(f, lineterminator='\n')  # 改行コード（\n）を指定しておく

        l = list(range(1, 43884 + 1))
        random.shuffle(l)
        for i in l[:nmax]:
            if i in [int(k[0]) for k in self.chords]:
                self.pbar.update(1)
                continue
            while 1:
                try:
                    writer.writerows([self.fetch(i)])
                    time.sleep(interval)
                    j = 0
                    break
                except Exception as e:
                    j += 1
                    time.sleep(2)
                    if j == 3:
                        print(e, '#error i=', i, 'です')
                        break
            if j == 3:
                break

        f.close()
        self.pbar.close()
            # writer.writerows(self.chords)  # 2次元配列も書き込める

    def fetch(self, i):
        title, hito, c = self.tochord(i)
        self.chords.append([i, title, hito, c])
        self.pbar.update(1)
        return [i, title, hito, c]


def main():
    file = 'data/chord.csv'
    site = ChordWiki(100)
    chords = []
    pbar = tqdm.tqdm(total=1000)
    if os.path.isfile(file):
        with open(file, 'r') as f:
            reader = csv.reader(f)
            chords = list(reader)
    for j in range(1, 1101, 100):
        sets = site.getlinks(j)
        for i in sets:
            if i[1] in [k[1] for k in chords]:
                pbar.update(1)
                continue
            title, chord = site.getchord(i[2])
            chord = site.tochord(chord)
            chords.append([i[0], title, chord])
            pbar.update(1)
            time.sleep(2)
    pbar.close()

    with open(file, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')  # 改行コード（\n）を指定しておく
        writer.writerows(chords)  # 2次元配列も書き込める


if __name__ == '__main__':
    UFRET().main()
    # main()