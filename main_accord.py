# %% 
import os
os.getcwd()
os.chdir("C:/Users/owner/OneDrive/ドキュメント/社会人博士/SReFT/script")

# %% Libraries
import warnings
import time
import math
import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sreft import SReFT
from sreft import DummyTransformer


# %% Functions
def n2mfrow(n_plots):
    n_plots = int(n_plots)
    if n_plots <= 3:
        return [n_plots, 1]
    if n_plots <= 6:
        return [(n_plots + 1) // 2, 2]
    if n_plots <= 12:
        return [(n_plots + 2) // 3, 3]
    if n_plots > 12:
        nrow = math.ceil(math.sqrt(n_plots))
        ncol = math.ceil(n_plots / nrow)
        do = True
        while(do and nrow * ncol > n_plots):
            if nrow * (ncol - 1) >= n_plots:
                ncol = ncol - 1
            elif (nrow - 1) * ncol >= n_plots:
                nrow = nrow - 1
            else:
                do = False
        return [nrow, ncol]

def calculate_mean_xy(x, y):
    if any(np.isnan(y).all(axis=1).all(axis=1)):
        raise Exception('all the values of y are nan for some samples')
    with warnings.catch_warnings() as w:
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        mean_x = np.where(np.isnan(y), np.nan, x)
        mean_x = np.nanmean(mean_x, axis=1, keepdims=True)
        mean_y = np.nanmean(y, axis=1, keepdims=True)
        mean_x = np.where(np.isnan(mean_x), np.nanmean(mean_x, axis=2, keepdims=True), mean_x)
        mean_y = np.where(np.isnan(mean_y), np.nanmean(mean_y, axis=0, keepdims=True), mean_y)

    mean_xy = np.concatenate([mean_x, mean_y], axis=-1)
    mean_xy = np.tile(mean_xy, (1, x.shape[1], 1))
    return mean_xy


# %%
data_accord = pd.read_csv('../data/ACCORD_SReFT_dataset_All2.csv').drop(columns='Visit')
# data_accord = data_accord[['MaskID', 'time', 'wt_kg', 'feelther', 'sbp', 'dbp', 'hr', 'hba1c', 'chol', 'trig', 'vldl', 'ldl', 'hdl', 'fpg', 'alt', 'cpk', 'potassium', 'screat', 'gfr', 'ualb', 'ucreat', 'uacr', 'HUI3Scor', 'HUI2pf']]
#使うバイオマーカーを選択。先頭のMaskID、2列目のtimeは変えない。それ以降はバイオマーカー。

name_biomarker = data_accord.columns.values[2:]
n_biomarker = len(name_biomarker)
n_obs = data_accord.groupby("MaskID").count().time.max()
n_subject = len(data_accord.MaskID.unique())
n_col = n2mfrow(n_biomarker)[1]
n_row = n2mfrow(n_biomarker)[0]

data_id = data_accord.groupby('MaskID')
x = np.empty([n_subject, n_obs, 1])
y = np.empty([n_subject, n_obs, n_biomarker])
y[:, :, :] = np.nan

for i, data in enumerate(data_id):
    dummy_x = data[1].time.values
    dummy_y = data[1].iloc[:, 2:].values
    adds = n_obs - len(dummy_x)
    if adds > 0:
        dummy_x = np.append(dummy_x, np.repeat(0, adds))
        dummy_y = np.vstack((dummy_y, np.tile(np.repeat(np.nan, n_biomarker), [adds, 1])))
    x[i, :, :] = dummy_x.reshape(1, n_obs, 1)
    y[i, :, :] = dummy_y.reshape(1, n_obs, n_biomarker)


# %%
m = calculate_mean_xy(x, y) #スケーリング（平均0、分散1）

scaler_x = DummyTransformer().fit(np.vstack(x))
scaler_m = StandardScaler().fit(np.vstack(m))
scaler_y = StandardScaler().fit(np.vstack(y))

x_scaled = scaler_x.transform(np.vstack(x)).reshape(x.shape)
m_scaled = scaler_m.transform(np.vstack(m)).reshape(m.shape)
y_scaled = scaler_y.transform(np.vstack(y)).reshape(y.shape)


# %%
class Logger(keras.callbacks.Callback):
    def set_params(self, params):
        self.epochs = params['epochs']
    def on_epoch_end(self, epoch, logs={}):
        self.last_epoch = epoch
        unit = [1, 10, 100][np.digitize(epoch, [10, 100])]
        if (epoch + 1) % unit == 0:
            self.__print(epoch, logs)
    def on_train_end(self, logs={}):
        self.__print(self.last_epoch, logs)
    def __print(self, epoch, logs={}):
        txt = ' - '.join([f'{k}: {v:9.3f}' for k, v in logs.items()])
        print(f'Epoch {epoch+1:5d}/{self.epochs} - {txt}')

_patience = 10

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=_patience, verbose=1,
    mode='min', baseline=None, restore_best_weights=True) #過学習

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=int(_patience/2), verbose=1,
    mode='min', min_delta=0.000, cooldown=0, min_lr=0) #最適化計算の刻み幅の調整

callbacks = [Logger(), earlystop, reduce_lr]


# %%
# 大元の解析。ほんちゃんのモデル構築をやっている
proc_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')[2:] + '_' #Figを作るとき用に日付情報を抽出しているだけ
m_train, m_test, x_train, x_test, y_train, y_test = train_test_split(m_scaled, x_scaled, y_scaled, test_size=0.1, random_state=1)

start = time.time()
sreft = SReFT(output_dim=n_biomarker, activation='tanh', latent_dim=128, random_state=1)
sreft.compile(optimizer=keras.optimizers.Adam(1e-3))
sreft.fit((m_train, x_train), y_train, batch_size=32, validation_data=((m_test, x_test), y_test), epochs=9999, verbose=0, callbacks=callbacks)
print('calculatin time (sec):', format(time.time() - start, '.2f'))


# %%
def prediction_plot(sreft, x_scaled, y_scaled, m_scaled, sup_title='', fig_name=''):
    x_data = x_scaled + sreft.model_1(m_scaled).numpy()
    y_data = scaler_y.inverse_transform(np.vstack(y_scaled)).reshape(y_scaled.shape)

    x_model = np.linspace(x_data.min(), x_data.max(), 100).reshape(100, 1)
    y_pred = sreft.model_y(x_model)
    y_pred = scaler_y.inverse_transform(np.vstack(y_pred)).reshape(y_pred.shape)

    fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 3, n_row * 3))
    for k, ax in enumerate(axs.flat):
        if k >= n_biomarker:
            ax.axis('off')
            continue

        ax.scatter(x_data, y_data[:, :, k], s=2, c='silver')
        ax.plot(x_model, y_pred[:, :, k], 'b-', linewidth=3)
        ax.set_title(name_biomarker[k], fontsize=15)
        ax.set_xlabel('Disease Time (year)')

    fig.tight_layout()
    fig.savefig('../output/' + proc_time + 'prediction.png', dpi=300, transparent=False)
    return None

prediction_plot(sreft, x_test, y_test, m_test)


# %%
def rowdata_plot(x_test, y_test):
    y_ = scaler_y.inverse_transform(np.vstack(y_test)).reshape(y_test.shape)
    fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 2, n_row * 3), sharey=True, tight_layout=True)
    for k, ax in enumerate(axs.flat):
        if k >= n_biomarker:
            ax.axis('off')
            continue

        for x, y in zip(x_test[:, :, 0], y_[:, :, k]):
            ax.plot(x[~np.isnan(y)], y[~np.isnan(y)], '.-', color='silver', linewidth=0.5)
            if is_random_biomarker[k] == 1:
                ax.set_title(f'Noise {k + 1}')
            else:
                ax.set_title(f'Biomarker {k + 1}')
            ax.set_xticks(np.arange(0, 5, 1))
            ax.set_xlabel('Observavtion Time (year)')
    # fig.savefig('./output/' + proc_time + 'rowdata.png', dpi=300, transparent=True)

rowdata_plot(x_test, y_test)


# %%
def compute_negative_log_likelihood(sreft, y_true, y_pred):
    neg_ll = sreft.lnvar_y + np.power(y_true - y_pred, 2) / np.exp(sreft.lnvar_y)
    return neg_ll

def compute_parmutation_importance(sreft, x_test, y_test, m_test, n_sample):
    rng = np.random.default_rng(42)

    y_pred = sreft((m_test, x_test)).numpy()
    neglls_orig = compute_negative_log_likelihood(sreft, y_test, y_pred)

    mean_scores = []
    std_scores = []
    for i in range(m_test.shape[2]):
        scores = []
        for j in range(n_sample):
            m_test_rand = np.copy(m_test)
            rng.shuffle(m_test_rand[:, :, i])

            y_pred_rand = sreft((m_test_rand, x_test)).numpy()
            neglls_rand = compute_negative_log_likelihood(sreft, y_test, y_pred_rand)
            nglls_diff = neglls_rand - neglls_orig
            negll_diff = np.nanmean(np.nansum(nglls_diff, axis=(1, 2)))

            scores.append(negll_diff)
        mean_scores.append(np.mean(scores))
        std_scores.append(np.std(scores))
    return mean_scores, std_scores

mean_scores, std_scores = compute_parmutation_importance(sreft, x_test, y_test, m_test, n_sample = 10)


# %%
plt.figure(figsize=(n_biomarker / 2, 8))
plt.bar(np.arange(n_biomarker) + 1, mean_scores[n_biomarker:n_biomarker * 2], yerr=std_scores[n_biomarker:n_biomarker * 2])
plt.xticks(np.arange(n_biomarker) + 1, name_biomarker, rotation=45)
plt.xlabel('meany')
plt.ylabel('Permutation Importance')
plt.savefig('../output/' + proc_time + 'pi_meany.png', dpi=300, transparent=False)

plt.figure(figsize=(n_biomarker, 8))
plt.bar(np.arange(n_biomarker * 2) + 1, mean_scores, yerr=std_scores)
plt.xticks(np.arange(n_biomarker * 2) + 1, np.tile(name_biomarker, 2), rotation=45)
plt.xlabel('meanx, meany')
plt.ylabel('Permutation Importance')
plt.savefig('../output/' + proc_time + 'pi_meanx_y.png', dpi=300, transparent=False)

plt.figure(figsize=(n_biomarker / 2, 8))
plt.bar(np.arange(n_biomarker) + 1, np.exp(sreft.lnvar_y))
plt.xticks(np.arange(n_biomarker) + 1, name_biomarker, rotation=45)
plt.ylabel('var_y')
plt.savefig('../output/' + proc_time + 'var_y.png', dpi=300, transparent=False)

# %% check tensorflow version
import sys

import tensorflow as tf
import tensorflow.keras

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print(f"Python {sys.version}")
gpu = len(tf.config.list_physical_device('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
