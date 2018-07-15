#!/usr/bin/env python

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

class RNN(chainer.Chain):
    def __init__(self):
        super(RNN, self).__init__()
        with self.init_scope():
            pass

class Word2Vec(chainer.Chain):
    def __init__(self, n_vocab, n_key, n_vec):
        super(Word2Vec, self).__init__()
        with self.init_scope():
            # self.embed = L.EmbedID(
            #     n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.head_encoder = L.EmbedID(n_vocab, n_vec)
            self.foot_encoder = L.EmbedID(n_vocab, n_vec)
            self.key_encoder = L.EmbedID(n_vocab, n_key)
            self.decoder = L.Linear(n_vocab, initialW=0)

        self.logreport = None

    def __call__(self, x):
        key = F.sum(self.key_encoder(x), axis=1) * (1. / x.shape[1])

        head = self.head_encoder(x[:, (x.shape[1]-1)//2])
        foot = self.foot_encoder(x[:, (x.shape[1]+1)//2])
        h = F.concat([head, foot, key])
        h = self.decoder(h)
        # y = (key+head+foot) / 3.0
        # e = self.head(around)
        # y = F.sum(e, axis=1) * (1. / x.shape[1])
        return h

    def lossfunc(self, x, t):
        h = self(x)

        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)
        r = loss_r(self.logreport.log, loss.data)

        chainer.reporter.report({'loss': loss}, self)
        chainer.reporter.report({'accuracy': accuracy}, self)
        if r:
            chainer.reporter.report({"loss_r": r}, self)
        return loss


class LSTM(chainer.Chain):
    def __init__(self, n_vocab, n_vec, n_layers, dropout=0.1):
        super(LSTM, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_vec)
            self.encoder = L.NStepLSTM(n_layers, n_vec, n_vocab, dropout)
            # self.decode = L.Linear(n_vocab)
        self.dropout = dropout
        self.logreport = None

    def __call__(self, x):
        h = self.sequence_embed(x)
        last_h, last_c, ys = self.encoder(None, None, h)
        h = last_h[-1]
        # h = self.decoder(h)
        return h

    def reset_state(self):
        self.lstm.reset_state()

    def lossfunc(self, x, t):
        h = self(x)
        t = F.concat(t, axis=0)

        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)
        r = loss_r(self.logreport.log, loss.data)

        chainer.reporter.report({'loss': loss}, self)
        chainer.reporter.report({'accuracy': accuracy}, self)
        if r:
            chainer.reporter.report({"loss_r": r}, self)
        return loss

    def sequence_embed(self, x):
        x_len = [len(i) for i in x]
        x_section = np.cumsum(x_len[:-1])
        ex = self.embed(F.concat(x, axis=0))
        ex = F.dropout(ex, ratio=self.dropout)
        exs = F.split_axis(ex, x_section, 0)
        return exs

def loss_r(log, loss, num=10):
    list_ = [i['main/loss'] for i in log[1-num:]] + [float(loss)]
    if len(list_) > 1:
        return np.corrcoef(range(len(list_)), list_)[0][1]
    return None


class Bottle_neck_block(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, activation=F.relu, init_stride=None, stride=1, pad=1):
        initializer = chainer.initializers.HeNormal()
        middle_channels = int(out_channels / 2)
        self.activation = activation
        super(Bottle_neck_block, self).__init__()
        with self.init_scope():
            # pre-activation & 参考: https://arxiv.org/pdf/1610.02915.pdf
            self.bn1 = L.BatchNormalization(in_channels)
            self.conv1 = L.ConvolutionND(1, in_channels, middle_channels, ksize=1, initialW=initializer)
            self.bn2 = L.BatchNormalization(middle_channels)
            self.conv2 = L.ConvolutionND(1, middle_channels, middle_channels, ksize, init_stride or stride, pad, initialW=initializer)
            self.bn3 = L.BatchNormalization(middle_channels)
            self.conv3 = L.ConvolutionND(1, middle_channels, out_channels, ksize=1, initialW=initializer)
            self.bn4 = L.BatchNormalization(out_channels)

            self.xconv = L.ConvolutionND(1, middle_channels, out_channels, ksize=1, stride=2, initialW=initializer)

    def __call__(self, x, ratio):
        h = self.bn1(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        h = self.conv2(h)
        h = self.activation(self.bn3(h))
        h = F.dropout(h, ratio)  # Stochastic Depth
        h = self.conv3(h)
        h = self.bn4(h)  # 必要?

        if x.shape[2:] != h.shape[2:]:  # skipではないほうのデータの縦×横がこのblock中で小さくなっていた場合skipの方もそれに合わせて小さくする
            #x = F.average_pooling_2d(x, 1, 2)  # これでいいのか？
            x = self.xconv(x)
        if x.shape[1] != h.shape[1]:  # skipではない方のデータのチャンネル数がこのblock内で増えている場合skipの方もそれに合わせて増やす(zero-padding)
            xp = chainer.cuda.get_array_module(x.data)  # GPUが使える場合も想定
            p = chainer.Variable(xp.zeros((x.shape[0], h.shape[1] - x.shape[1], *x.shape[2:]), dtype=xp.float32))
            x = F.concat((x, p))
        return x + h


class Generator(chainer.Chain):
    def __init__(self, n_voc, n_hidden):
        super(Generator, self).__init__()
        self.n_voc = n_voc
        self.n_hidden = n_hidden
        initializer = chainer.initializers.HeNormal()
        with self.init_scope():
            self.l = L.Linear(450*15)

            self.conv1 = L.ConvolutionND(1, 16, 32, 7, stride=1, pad=3)

            self.block1_1 = Bottle_neck_block(32, 32, 3, pad=1)
            self.block1_2 = Bottle_neck_block(32, 32, 3, pad=1)

            self.block2_1 = Bottle_neck_block(32, 64, 3, pad=1)
            self.block2_2 = Bottle_neck_block(64, 64, 3, pad=1)

            self.block3_1 = Bottle_neck_block(64, 128, 3, pad=1)
            self.block3_2 = Bottle_neck_block(128, 128, 3, pad=1)


            # self.dc1 = L.DeconvolutionND(1, 32, 32, ksize=7, stride=1)
            # self.dc2 = L.DeconvolutionND(1, 32, 32, ksize=5, stride=1)
            # self.dc3 = L.DeconvolutionND(1, 32, 32, ksize=3, stride=1)
            # self.dc4 = L.DeconvolutionND(1, 32, 32, ksize=3, stride=1)
            # self.dc5 = L.DeconvolutionND(1, 32, 32, ksize=3, stride=1)
            self.decoder = L.ConvolutionND(1, 128, n_voc, ksize=1, stride=1)
            #
            # # self.bn0 = L.BatchNormalization(size=self.bottom_width*self.bottom_width*self.ch)
            # self.bn1 = L.BatchNormalization(32)
            # self.bn2 = L.BatchNormalization(32)
            # self.bn3 = L.BatchNormalization(32)
            # self.bn4 = L.BatchNormalization(32)
            # self.bn5 = L.BatchNormalization(32)

    def make_noise(self, batchsize):
        return np.random.uniform(-1, 1, (batchsize, self.n_hidden)).astype(np.float32)

    def __call__(self, z):
        h = self.l(z)
        h = F.reshape(h, [len(h), 15, 450])
        ren = chainer.Variable(np.array([[[i for i in range(1, 1 + h.shape[2])]] * 1] * h.shape[0], dtype=np.float32))
        h = F.concat([h, ren/450], axis=1)

        n = 0.5 / 6
        h = self.conv1(h)
        h = self.block1_1(h, 1 * n)  # => 64  ×  32
        h = self.block1_2(h, 2 * n)  # => 64  ×  32

        h = self.block2_1(h, 3 * n)  # => 128 ×  16
        h = self.block2_2(h, 4 * n)  # => 128 ×  16

        h = self.block3_1(h, 5 * n)  # => 256 ×   8
        h = self.block3_2(h, 6 * n)  # => 256 ×   8


        # h = F.relu(self.bn1(self.dc1(h)))
        # h = F.relu(self.bn2(self.dc2(h)))
        # h = F.relu(self.bn3(self.dc3(h)))
        # # h = F.relu(self.bn4(self.dc4(h)))
        # # h = F.relu(self.bn5(self.dc5(h)))
        h = self.decoder(h)
        return h


class Discriminator(chainer.Chain):
    def __init__(self, n_vocab, n_vec, n_layers=2, bottom_width=3, ch=512, wscale=0.02):
        initializer = chainer.initializers.HeNormal()
        self.dropout = 0.1
        self.n_vec = n_vec
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_vec)
            self.encoder = L.NStepBiLSTM(n_layers, n_vec, n_vec, self.dropout)

            self.block1_1 = Bottle_neck_block(32, 32, 3, activation=F.leaky_relu, pad=1)
            self.block1_2 = Bottle_neck_block(32, 32, 3, activation=F.leaky_relu, pad=1)

            self.block2_1 = Bottle_neck_block(32, 64, 3, activation=F.leaky_relu, init_stride=2, pad=1)
            self.block2_2 = Bottle_neck_block(64, 64, 3, activation=F.leaky_relu, pad=1)

            self.block3_1 = Bottle_neck_block(64, 128, 3, activation=F.leaky_relu, init_stride=2, pad=1)
            self.block3_2 = Bottle_neck_block(128, 128, 3, activation=F.leaky_relu, pad=1)

            self.l = L.Linear(2, initialW=initializer)


    def __call__(self, x):
        if isinstance(x, chainer.Variable):
            h = F.softmax(x, axis=1)
            w = F.expand_dims(self.embed.W.T, axis=2)
            h = F.convolution_nd(h, w)
            h += F.transpose(self.embed(F.argmax(x, axis=1)), [0, 2, 1])
            h /= 2
        else:
            h = self.sequence_embed(x)

        n = 0.5 / 6
        h = self.block1_1(h, 1 * n)  # => 64  ×  32
        h = self.block1_2(h, 2 * n)  # => 64  ×  32

        h = self.block2_1(h, 3 * n)  # => 128 ×  16
        h = self.block2_2(h, 4 * n)  # => 128 ×  16

        h = self.block3_1(h, 5 * n)  # => 256 ×   8
        h = self.block3_2(h, 6 * n)  # => 256 ×   8

        # h = F.average_pooling_nd(h, h.shape[2])  # global average pooling
        h = F.spatial_pyramid_pooling_2d(F.expand_dims(h, axis=3), 2, F.MaxPooling2D)
        h = F.reshape(h, h.shape[:2])

        return self.l(h)

    def sequence_embed(self, x):
        x_len = [len(i) for i in x]
        # x_section = np.cumsum(x_len[:-1])
        ex = F.pad_sequence(x)
        ex = self.embed(F.concat(ex, axis=0))
        ex = F.dropout(ex, ratio=self.dropout)
        # exs = F.split_axis(ex, [max(x_len)*i for i in range(1, len(x)+1)] * len(x), 0)
        exs = F.reshape(ex, [len(x), max(x_len), self.n_vec])
        return F.transpose(exs, [0, 2, 1])
