#!/usr/bin/env python

import json
import argparse
import copy

import chainer
from chainer import training
from chainer.training import extensions
import mymodel
import chord
import csv
import numpy as np
import os


class MyEvaluator(extensions.Evaluator):
    """
    chainer標準のEvaliator(testデータの評価出力)で
    取るべきでないのに各バッチの平均が取られてしまう問題を修正し
    省略したPrintReportに対応したクラス
    """
    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = chainer.reporter.DictSummary()

        for batch in it:
            observation = {}
            with chainer.reporter.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with chainer.function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)

            summary.add(observation)

        d = summary.compute_mean()
        return {'val/' + name.split('/')[-1]: sammary for name, sammary in d.items()}


class Trans(object):
    def __init__(self, window):
        self.windowsize = window
        file = 'data/ufret_emb.csv'

        if not os.path.isfile(file):
            self.shot = chord.oneshot()
        else:
            with open(file, 'r') as f:
                reader = csv.reader(f)
                self.shot = list(reader)
                self.shot = [[int(j) for j in i] for i in self.shot]

    def getindex(self):
        return [(i, k) for i, j in enumerate(self.shot) for k in range(len(j))]

    def getchord(self, i, j):
        start = max(0, j-self.windowsize)
        head = self.shot[i][start:j]
        head = [0] * (self.windowsize-len(head)) + head

        # end = min(len(self.shot[i]), j+self.windowsize+1)
        foot = self.shot[i][j+1:j+self.windowsize+1]
        foot = foot + [0] * (self.windowsize-len(foot))

        return np.concatenate([head, foot]).astype(np.int32), np.array(self.shot[i][j], dtype=np.int32)

    def __call__(self, i):
        x, t = self.getchord(*i)
        return x, t


def main():
    parser = argparse.ArgumentParser(
        description='RNNとかで曲生成したい!')
    parser.add_argument('--batchsize', '-b', type=int, default=1024,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--interval', '-i', type=int, default=100,
                        help='プログレスバー,表示とかのインターバル')
    parser.add_argument('--key', '-k', type=int, default=12,
                        help='中間層の次元')
    parser.add_argument('--vec', '-v', type=int, default=24,
                        help='中間層の次元')
    parser.add_argument('--window', '-w', type=int, default=4,
                        help='単語の左右の窓幅')
    parser.add_argument('--model', '-model', default='Word2Vec',
                        choices=['Word2Vec'],
                        help='Name of encoder model type.')

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    model = getattr(mymodel, args.model)(481, args.key, args.vec)

    # GPUで動かせるのならば動かす
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # optimizerのセットアップ
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # データセットのセットアップ
    trans = Trans(args.window)
    index = trans.getindex()
    train, val = chainer.datasets.split_dataset_random(
        index, int(len(index) * 0.8), seed=0)  # 2割をvalidation用にとっておく
    train = chainer.datasets.TransformDataset(train, trans)
    val = chainer.datasets.TransformDataset(val, trans)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(val, args.batchsize,
                                                repeat=False, shuffle=False)

    # 学習をどこまで行うかの設定
    stop_trigger = (args.epoch, 'epoch')

    # uodater, trainerのセットアップ
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu, loss_func=model.lossfunc)
    trainer = training.Trainer(updater, stop_trigger)

    # testデータでの評価の設定
    evaluator = MyEvaluator(val_iter, model, device=args.gpu, eval_func=model.lossfunc)
    evaluator.trigger = 1, 'epoch'
    # trainer.extend(evaluator)

    # モデルの層をdotファイルとして出力する設定
    trainer.extend(extensions.dump_graph('main/loss'))

    # snapshot(学習中の重み情報)の保存
    frequency = 1
    # trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # trainデータでの評価の表示頻度設定
    logreport = extensions.LogReport(trigger=(args.interval, 'iteration'))
    trainer.extend(logreport)
    model.logreport = logreport

    # 各データでの評価の保存設定
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'val/loss'],
                'iteration', trigger=(100, 'iteration'), file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/acc', 'val/acc'],
                'iteration', trigger=(100, 'iteration'), file_name='accuracy.png'))

    # 各データでの評価の表示(欄に関する)設定
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss_r', 'main/loss', 'val/loss', 'main/acc', 'elapsed_time']))

    # プログレスバー表示の設定
    trainer.extend(extensions.ProgressBar(update_interval=args.interval))

    # 学習済みデータの読み込み設定
    # if args.resume:
    #     chainer.serializers.load_npz(args.resume, model, path='updater/model:main/')  # なぜかpathを外すと読み込めなくなってしまった 原因不明

    trainer.run()
    # Save the word2vec model
    # with open(str(args.unit * 3) + 'dword2vec.model', 'w') as f:
    with open('ufret-word2vec.model', 'w') as f:
        w1 = chainer.cuda.to_cpu(model.head.W.data)
        w2 = chainer.cuda.to_cpu(model.foot.W.data)
        w3 = chainer.cuda.to_cpu(model.key.W.data)
        w = np.concatenate([w1, w2, w3], axis=1)
        for i, wi in enumerate(w):
            v = '	'.join(map(str, wi))
            f.write('%s\n' % v)


if __name__ == '__main__':
    main()
