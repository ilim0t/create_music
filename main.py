#!/usr/bin/env python

import json
import argparse
import six
import copy
import pretty_midi

import chainer
from chainer import training
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L

import mymodel


class Code(object):
    #  1  3     6  8  10    13 15    18
    #  D- E-    G- A- B-    D- E-    G-
    # C  D  E  F  G  A  B  C  D  E  F  G
    # 0  2  4  5  7  9  11 12 14 16 17 19
    code_notes = {
        'X': [0, 4, 7],
        'Xm': [0, 3, 7],
        'Xaug': [0, 4, 8],
        'Xdim': [0, 3, 6],
        'Xsus4': [0, 5, 7],
        'Xsus2': [0, 2, 7],
        'X6': [0, 4, 9],
        'Xm6': [0, 3, 9],
    }
    code_notes.update({
        'X7': code_notes['X'] + [10],
        'XM7': code_notes['X'] + [11],
        'Xm7': code_notes['Xm'] + [10],
        'XmM7': code_notes['Xm'] + [11],
        'Xm7-5': code_notes['Xdim'] + [10],
        'X7sus4': code_notes['Xsus4'] + [10],
        'Xdim7': code_notes['Xdim'] + [9],
        'Xaug7': code_notes['Xaug'] + [11]
    })
    def to_note(self, code):




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
        freq_errs = dict()

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
            if 'validation/main/freq_err' in observation.keys():
                freq_errs[observation['validation/main/freq_err']] = \
                    freq_errs.get(observation['validation/main/freq_err'], 0) + 1

        d = {name: summary.compute_mean() for name, summary in six.iteritems(summary._summaries)}
        if 'validation/main/freq_err' in observation.keys():
            d['validation/main/freq_err'] = max([(v, k) for k, v in freq_errs.items()])[1]
        return {'val/' + name.split('/')[-1]: sammary for name, sammary in d.items()}



def main():
    parser = argparse.ArgumentParser(
        description='RNNとかで曲生成したい!')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-model', default='rnn',
                        choices=['rnn'],
                        help='Name of encoder model type.')

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    model = getattr(mymodel, args.model)(args.label_variety, args.lossfunc)

    # GPUで動かせるのならば動かす
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # optimizerのセットアップ
    optimizer = chainer.optimizers.Adam()
    # optimizer = chainer.optimizers.MomentumSGD(0.1, 0.9)  # https://arxiv.org/pdf/1605.07146.pdf
    # chainer.optimizer.WeightDecay(0.0005)

    optimizer.setup(model)

    # データセットのセットアップ
    photo_nums = 1#photos(args)
    train, val = chainer.datasets.split_dataset_random(
        photo_nums, int(len(photo_nums) * 0.8), seed=0)  # 2割をvalidation用にとっておく
    trans = lambda x: x#Transform(args, photo_nums, True, False if args.model == 5 else True)
    train = chainer.datasets.TransformDataset(train, trans)
    val = chainer.datasets.TransformDataset(val, trans)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(val, args.batchsize,
                                                repeat=False, shuffle=False)

    # 学習をどこまで行うかの設定
    stop_trigger = (args.epoch, 'epoch')
    # if args.early_stopping:  # optimizerがAdamだと無意味
    #     stop_trigger = training.triggers.EarlyStoppingTrigger(
    #         monitor=args.early_stopping, verbose=True,
    #         max_trigger=(args.epoch, 'epoch'))

    # uodater, trainerのセットアップ
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu, loss_func=model.loss_func)
    trainer = training.Trainer(updater, stop_trigger, out=args.out)

    # testデータでの評価の設定
    evaluator = MyEvaluator(val_iter, model, device=args.gpu, eval_func=model.loss_func)
    evaluator.trigger = 1, 'epoch'
    trainer.extend(evaluator)

    # 学習済み部分を学習しないように (backwardはされてるっぽい?)
    if args.model == 6 or args.model == 7:
        model.base.disable_update()

    # モデルの層をdotファイルとして出力する設定
    trainer.extend(extensions.dump_graph('main/loss'))

    # snapshot(学習中の重み情報)の保存
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # trainデータでの評価の表示頻度設定
    logreport = extensions.LogReport(trigger=(args.interval, 'iteration'))
    trainer.extend(logreport)

    # 各データでの評価の保存設定
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'val/loss'],
                'iteration', trigger=(5, 'iteration'), file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/acc', 'val/acc'],
                'iteration', trigger=(5, 'iteration'), file_name='accuracy.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/freq_err', 'val/freq_err'],
                'iteration', trigger=(5, 'iteration'), file_name='frequent_error.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/acc2', 'val/acc2'],
                'iteration', trigger=(5, 'iteration'), file_name='accuracy2.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/f1', 'val/f1'],
                'iteration', trigger=(5, 'iteration'), file_name='f1.png'))

    # 各データでの評価の表示(欄に関する)設定
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'val/loss', 'main/acc', 'main/acc2', 'val/acc2',
         'main/precision', 'main/recall', 'main/f1', 'val/f1', 'main/labelnum', 'main/fpk', 'elapsed_time']))

    # プログレスバー表示の設定
    trainer.extend(extensions.ProgressBar(update_interval=args.interval))

    # SGD用 学習率調整設定
    # trainer.extend(MyShift("lr", 1 / 5, logreport, 0.1))

    # 学習済みデータの読み込み設定
    if args.resume:
        chainer.serializers.load_npz(args.resume, model, path='updater/model:main/')  # なぜかpathを外すと読み込めなくなってしまった 原因不明

    trainer.run()


if __name__ == '__main__':
    main()
