from pydub import AudioSegment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sound = AudioSegment.from_file("data/シャルル.mp3", "mp3")

samples = np.array(sound.get_array_of_samples())
sample = samples[::sound.channels]

#スペクトル格納用
ampList = []
#偏角格納用
argList = []

#窓幅
w = 1000
#刻み
s = 500

#刻みずつずらしながら窓幅分のデータをフーリエ変換する
for i in range(int((sample.shape[0]- w) / s)):
    data = sample[i*s:i*s+w]
    spec = np.fft.fft(data)
    spec = spec[:int(spec.shape[0]/2)]
    spec[0] = spec[0] / 2
    ampList.append(np.abs(spec))
    argList.append(np.angle(spec))

#周波数は共通なので１回だけ計算（縦軸表示に使う）
freq = np.fft.fftfreq(data.shape[0], 1.0/sound.frame_rate)
freq = freq[:int(freq.shape[0]/2)]

#時間も共通なので１回だけ計算（横軸表示に使う）
time = np.arange(0, i+1, 1) * s / sound.frame_rate

#numpyの配列にしておく
ampList = np.array(ampList)
argList = np.array(argList)

df_amp = pd.DataFrame(data=ampList, index=time, columns=freq)

#seabornのheatmapを使う
plt.figure(figsize=(20, 6))
sns.heatmap(data=df_amp.iloc[:, :100].T,
            xticklabels=100,
            yticklabels=10,
            cmap=plt.cm.gist_rainbow_r,
           )
plt.show()
while 1:
    eval(input())