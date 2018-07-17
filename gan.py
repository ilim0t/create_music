#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import copy

import chainer
from chainer import training
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F

import mymodel
import chord
import csv
import numpy as np
import os
import mymodel
from chord import tochord

from nlp_utils import convert_seq


class Trans(object):
    def __init__(self):
        file = 'data/ufret_emb.csv'

        if not os.path.isfile(file):
            self.shot = chord.oneshot()
        else:
            with open(file, 'r') as f:
                reader = csv.reader(f)
                self.shot = list(reader)
                self.shot = [[int(j) for j in i] for i in self.shot]
        # self.lenshot = {}
        # for i in self.shot:
        #     self.lenshot[len(i)] = self.lenshot.get(len(i), []) + [i]

    def getkeys(self):
        # return list(self.lenshot.keys())
        return list(range(len(self.shot)))

    def __call__(self, key):
        # return np.array(self.lenshot[key], dtype=np.int32)
        return np.array(self.shot[key], dtype=np.int32)


class DCGANUpdater(chainer.training.StandardUpdater):
    # 0 -> 生成由来
    # 1 -> 実際由来
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super(DCGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        # loss = -F.sum(F.log_softmax(y_real)[:, 0]) / batchsize
        # loss += -F.sum(F.log_softmax(y_fake)[:, 1]) / batchsize
        # loss /= 2
        chainer.report({'real_loss': L1}, dis)
        chainer.report({'fake_loss': L2}, dis)
        chainer.report({'loss': loss}, dis)

        precision = np.sum(y_real.data>0) / (np.sum(y_fake.data>0) + np.sum(y_real.data>0))
        recall = np.average(y_real.data>0)
        chainer.report({'precision': precision}, dis)
        chainer.report({'recall': recall}, dis)

        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        # loss = -F.sum(F.log_softmax(y_fake)[:, 0]) / batchsize
        chainer.report({'loss': loss}, gen)

        miss = np.average(y_fake.data>0)
        chainer.report({'miss': miss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        x_real = self.converter(batch, self.device)
        xp = chainer.cuda.get_array_module(x_real[0].data)
        gen, dis = self.gen, self.dis
        batchsize = len(batch)

        y_real = dis(x_real)
        z = chainer.Variable(xp.asarray(gen.make_noise(batchsize)))
        x_fake = gen(z)
        y_fake = dis(x_fake)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)

def my_converter(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = chainer.cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x)
                                     for x in batch[:-1]], dtype=np.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = chainer.cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return to_device_batch([x for x in batch])


def show_chord(gen, dis, num, seed):
    @chainer.training.make_extension(trigger=(1, 'epoch'))
    def make_chord(trainer):
        np.random.seed(seed)
        xp = gen.xp
        z = chainer.Variable(xp.asarray(gen.make_noise(num)))

        with chainer.using_config('train', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        # y = F.argmax(x, axis=1)
        # y = [[tochord(j) for j in i]for i in y.data]
        a = dis.embed
        with open('result/chord.txt', 'w') as f:
            for i in x:
                y = [tochord(j) for j in dis.embed.reverse(i.transpose([1,0]))]
                f.write("   ".join(y) + "\n")
    return make_chord

def main():
    parser = argparse.ArgumentParser(
        description='RNNとかで曲生成したい!')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--interval', '-i', type=int, default=2,
                        help='プログレスバー,表示とかのインターバル')
    parser.add_argument('--vec', '-v', type=int, default=64,
                        help='中間層の次元')
    parser.add_argument('--layer', '-l', type=int, default=2,
                        help='レイヤーの層')
    parser.add_argument('--frequency', '-f', type=int, default=1,
                        help='保存頻度')

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))


    gen = mymodel.Generator(481, 32)
    dis = mymodel.Discriminator(481, args.vec)

    # GPUで動かせるのならば動かす
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    # optimizerのセットアップ
    opt_gen = chainer.optimizers.Adam()
    opt_dis = chainer.optimizers.Adam()
    opt_gen.setup(gen)
    opt_dis.setup(dis)
    opt_gen.add_hook(chainer.optimizer.WeightDecay(0.0001))
    opt_dis.add_hook(chainer.optimizer.WeightDecay(0.0001))

    # データセットのセットアップ
    trans = Trans()
    index = trans.getkeys()
    # train, _ = chainer.datasets.split_dataset_random(
    #     index, int(len(index) * 0.8), seed=0)  # 2割をvalidation用にとっておく
    train = index
    train = chainer.datasets.TransformDataset(train, trans)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # 学習をどこまで行うかの設定
    stop_trigger = (args.epoch, 'epoch')

    # uodater, trainerのセットアップ
    updater = DCGANUpdater(models=(gen, dis), iterator=train_iter, optimizer={'gen': opt_gen, 'dis': opt_dis},
                           converter=my_converter, device=args.gpu)
    trainer = training.Trainer(updater, stop_trigger)

    # snapshot(学習中の重み情報)の保存
    frequency = (args.frequency, 'epoch')
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'), trigger=frequency)
    trainer.extend(extensions.snapshot_object(gen, 'gen_epoch_{.updater.epoch}.npz'), trigger=frequency)
    trainer.extend(extensions.snapshot_object(dis, 'dis_epoch_{.updater.epoch}.npz'), trigger=frequency)

    # trainデータでの評価の表示頻度設定
    logreport = extensions.LogReport(trigger=(args.interval, 'iteration'))
    trainer.extend(logreport)
    # model.logreport = logreport

    trainer.extend(show_chord(gen, dis, 10, seed=0), trigger=(args.interval, 'iteration'))

    # 各データでの評価の保存設定
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['gen/loss', 'dis/real_loss', 'dis/fake_loss'],
                'iteration', trigger=(10, 'iteration'), file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['gen/miss', 'dis/precision', 'dis/recall'],
                'iteration', trigger=(10, 'iteration'), file_name='accuracy.png'))

    # 各データでの評価の表示(欄に関する)設定
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'gen/loss', 'dis/real_loss', 'dis/fake_loss', 'dis/loss',
         'gen/miss', 'dis/precision', 'dis/recall', 'elapsed_time']))

    # プログレスバー表示の設定
    trainer.extend(extensions.ProgressBar(update_interval=args.interval))

    trainer.run()

if __name__ == '__main__':
    main()
