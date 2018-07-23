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


class EmbedID(L.EmbedID):

    # vec: Numpy / Cupy
    def cosine_similarity(self, vec):
        W = self.W.data
        xp = chainer.cuda.get_array_module(*(vec,))
        w_norm = xp.sqrt(xp.sum(W ** 2, axis=1))
        v_norm = xp.sqrt(xp.sum(vec ** 2, axis=1))
        inner_product = W.dot(vec.T)
        norm = w_norm.reshape(1, -1).T.dot(v_norm.reshape(1, -1)) + 1e-6
        # 最初の軸がIDに対応
        return inner_product / norm

    # vec: Numpy / Cupy
    def reverse(self, vec, sample=False, to_cpu=False):
        xp = chainer.cuda.get_array_module(*(vec,))
        if sample:
            result = self.reverse_sampling(vec)  # Numpy ndarray
            if to_cpu or xp is np:
                return result
            return chainer.cuda.to_cpu(result)
        else:
            # 最初の軸がIDに対応
            result = self.reverse_argmax(vec)  # Numpy ndarray or Cupy ndarray
            if to_cpu and xp is chainer.cuda.cupy:
                result = chainer.cuda.to_cpu(result)
            return result

    # vec: Numpy / Cupy
    # Returns xp
    def reverse_argmax(self, vec):
        xp = chainer.cuda.get_array_module(*(vec,))
        cos = self.cosine_similarity(vec)
        # 最初の軸がIDに対応する
        return xp.argmax(cos, axis=0)

    # vec: Numpy / Cupy
    # Returns np
    def reverse_sampling(self, vec):
        xp = chainer.cuda.get_array_module(*(vec,))
        cos = self.cosine_similarity(vec)
        cos = xp.exp(cos)
        sum = xp.sum(cos, axis=0)
        sum = sum.reshape(1, -1)
        softmax = cos / sum
        if xp is chainer.cuda.cupy:
            softmax = chainer.cuda.to_cpu(softmax)
        softmax = softmax.T
        n_vec = softmax.shape[0]
        n_ids = softmax.shape[1]
        result = np.empty((n_vec,), dtype=np.int32)
        for t in range(n_vec):
            id = np.random.choice(np.arange(n_ids), p=softmax[t])
            result[t] = id
        return result


class GenBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride, pad):
        super(GenBlock, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(in_channels)
            self.deconv = L.DeconvolutionND(1, in_channels, out_channels, ksize, stride, pad)

    def __call__(self, h):
        h = self.bn(h)
        h = self.deconv(h)
        return h


class Generator(chainer.Chain):
    def __init__(self, n_voc, n_hidden):
        super(Generator, self).__init__()
        self.n_voc = n_voc
        self.n_hidden = n_hidden
        initializer = chainer.initializers.HeNormal()
        with self.init_scope():
            self.l = L.Linear(16*16)
            self.deconv = L.DeconvolutionND(1, 16, 32, 7, stride=2, pad=2)

            self.block1 = GenBlock(32, 32, 4, 2, 1)
            self.block2 = GenBlock(32, 64, 4, 2, 1)
            self.block3 = GenBlock(64, 64, 4, 2, 1)
            self.block4 = GenBlock(64, 128, 4, 2, 1)
            self.block5 = DisBlock(128, 128, 5, 2, 0)
            self.block6 = DisBlock(128, 256, 3, 1, 1)

            self.bn = L.BatchNormalization(256)
            self.decoder = L.ConvolutionND(1, 256, n_voc, ksize=1, stride=1)

    def make_noise(self, batchsize):
        return np.random.uniform(-1, 1, (batchsize, self.n_hidden)).astype(np.float32)

    def __call__(self, z):  # 32
        h = self.l(z)  # 1024
        h = F.reshape(h, [h.shape[0], 16, 16])  ## 64, 16

        h = self.deconv(h)  # 128, 33

        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        # h = self.block7(h)
        # h = self.block8(h)
        h = self.bn(h)
        h = self.decoder(h)
        return h


class DisBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride, pad):
        super(DisBlock, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(in_channels)
            self.conv = L.ConvolutionND(1, in_channels, out_channels, ksize, stride, pad)

    def __call__(self, h):
        h = self.bn(h)
        h = self.conv(h)
        return h

class Discriminator(chainer.Chain):
    def __init__(self, n_vocab, n_vec, n_layers=2, bottom_width=3, ch=512, wscale=0.02):
        initializer = chainer.initializers.HeNormal()
        self.dropout = 0.1
        self.n_vec = n_vec
        super(Discriminator, self).__init__()
        with self.init_scope():
            # self.embed = L.EmbedID(n_vocab, n_vec)
            self.embed = EmbedID(n_vocab, n_vec)

            self.bn1 = L.BatchNormalization(n_vec)
            self.conv1 = L.ConvolutionND(1, n_vec, 32, 7, stride=2, pad=3)

            self.block1 = DisBlock(32, 32, 3, 1, 0)
            self.block2 = DisBlock(32, 64, 3, 2, 0)
            self.block3 = DisBlock(64, 64, 3, 1, 0)
            self.block4 = DisBlock(64, 128, 3, 2, 0)
            self.block5 = DisBlock(128, 128, 3, 1, 0)
            self.block6 = DisBlock(128, 256, 3, 2, 0)

            self.bn2 = L.BatchNormalization(256)
            self.l = L.Linear(1, initialW=initializer)


    def __call__(self, x):
        if isinstance(x, chainer.Variable):
            # h = self.bn1(x)
            h = F.softmax(x, axis=1)
            w = F.expand_dims(self.embed.W.T, axis=2)
            h = F.convolution_nd(h, w)
            # # h = F.transpose(self.embed(F.argmax(x, axis=1)), [0, 2, 1])
            # # h /= 2
        else:
            h = self.sequence_embed(x)

        h = self.bn1(h)
        h = self.conv1(h)

        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        # h = self.block7(h)
        # h = self.block8(h)

        h = self.bn2(h)
        h = F.average_pooling_nd(h, h.shape[2])  # global average pooling
        h = self.l(h)  # 正: 正解由来
        h = F.sigmoid(h)
        return h

    def sequence_embed(self, x):
        x_len = [len(i) for i in x]
        # x_section = np.cumsum(x_len[:-1])
        ex = F.pad_sequence(x)
        ex = self.embed(F.concat(ex, axis=0))
        ex = F.dropout(ex, ratio=self.dropout)
        # exs = F.split_axis(ex, [max(x_len)*i for i in range(1, len(x)+1)] * len(x), 0)
        exs = F.reshape(ex, [len(x), max(x_len), self.n_vec])
        return F.transpose(exs, [0, 2, 1])
