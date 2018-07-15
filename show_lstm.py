#!/usr/bin/env python

import chainer
import numpy as np
import chord
from pychord.constants import QUALITY_DICT, NOTE_VAL_DICT, SCALE_VAL_DICT
import mymodel
import json
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='RNNとかで曲生成したい!')
    parser.add_argument('--resume', '-r', type=str, default="result/lstm/snapshot_iter_900",
                        help='保存済みデータの名前')
    parser.add_argument('--vec', '-v', type=int, default=32,
                        help='中間層の次元')
    parser.add_argument('--layer', '-l', type=int, default=2,
                        help='レイヤーの層')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-model', default='LSTM',
                        choices=['Word2Vec'],
                        help='Name of encoder model type.')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    model = getattr(mymodel, args.model)(481, args.vec, args.layer)

    # GPUで動かせるのならば動かす
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # 学習済みデータの読み込み設定
    if args.resume:
        try:
            chainer.serializers.load_npz(args.resume, model, path='updater/model:main/')  # なぜかpathを外すと読み込めなくなってしまった 原因不明
        except Exception as e:
            print(e)
            chainer.serializers.load_npz(args.resume, model)  # なぜかpathを外すと読み込めなくなってしまった 原因不明

    while 1:
        try:
            str_ = input("\n何を予測する？ 空白区切りで入れてください\n")
            if str_ == "":
                break
            strs = str_.split(" ")
            if strs[-1] == "":
                strs = strs[:-1]
            chords = []
            for i in strs:
                c = chord.Chord(i)
                note = 1 + len(QUALITY_DICT) * NOTE_VAL_DICT[c.root] + \
                dict([(j, i) for i, j in list(enumerate(QUALITY_DICT.keys()))])[c.quality.quality]
                chords.append(note)
            indata = np.reshape(np.array(chords, dtype=np.int32), [1, len(chords)])

            y = model(indata)
            y = y.data[0]
            y = list(enumerate(y))
            y.sort(key=lambda x: x[1])
            y.reverse()
            y = [(chord.tochord(i[0]), i[1]) for i in y[:10]]
            for num, i in enumerate(y):
                print("No.{:<2} {:<4}: {}".format(num+1, i[0], i[1]))
        except Exception as e:
            print("Error")
            print(e)


if __name__ == "__main__":
    main()