
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
#from plot import plot_loss
#from time_history import TimeHistory, total_time
from matplotlib import pyplot as plt
import gc  # Python garbage collect
from tools import *
from datetime import datetime
train_file = 'train_2.csv'
input_df = pd.read_csv(train_file, nrows=1)
input_df_dates = input_df.columns[1:]; 
visits_dtype = {d: np.float32 for d in input_df_dates}
print('%%% Reading data '+ train_file + ' ... ', end = '', flush = True)
input_df = pd.read_csv( train_file, engine='c', dtype=visits_dtype)
print('done!')
#all_dataSet = np.load('train_2.csv')
def weekday(datestr):
    return datetime.strptime(datestr,'%Y-%m-%d').weekday()
def daydiff(dstr1,dstr2):
    return datetime.strptime(dstr1,'%Y-%m-%d') - datetime.strptime(dstr2,'%Y-%m-%d')

x_length = 64  # input period
y_length = 64  # predict period
test_length = 0  # for predicting
X_input_dates = input_df_dates[-x_length-test_length-364:-test_length-364]

if test_length:
    X_output_dates = input_df_dates[-x_length-test_length:-test_length]
else:
    X_output_dates = input_df_dates[-x_length:]


if test_length == 0:  # Prefetch the key_file
    output_df = pd.read_csv( 'key_2.csv', nrows=100)    # Read the first 100 days of required submission
    output_df['date'] = output_df.Page.apply(lambda a: a[-10:])  # take the last 10 characters from 'Page' as date
    output_df['Page'] = output_df.Page.apply(lambda a: a[:-11])  # remove the last 10 caharacters from 'Page'
    print(output_df.head())


if (test_length == 0):
    output_df_dates_all = output_df.date.values.astype('datetime64[D]')
    output_df_first_day = str(output_df_dates_all.min())
    output_df_final_day = str(output_df_dates_all.max())
    print('submission required first day:', output_df_first_day)
    print('submission required final day:', output_df_final_day)
Y_input_dates = input_df_dates[-test_length-364:-test_length-364+y_length]
# Y for output set
if test_length:
    Y_output_dates = input_df_dates[-test_length:]
else:
    Y_output_first_day = '2017-09-11'  ## Make sure it is correct when predicting !!!!!!!
    Y_output_dates = pd.Index(np.arange(np.datetime64(Y_output_first_day), 
                                        np.datetime64(Y_output_first_day)
                                        + np.timedelta64(y_length, 'D')).astype('str'))
print('Y_input_first_day: ', Y_input_dates[0], weekday(Y_input_dates[0]))
print('Y_input_final_day: ', Y_input_dates[-1], weekday(Y_input_dates[-1]))
print('Y_input_days_diff: ', daydiff(Y_input_dates[-1], Y_input_dates[0]))

print('Y_output_first_day:', Y_output_dates[0], weekday(Y_output_dates[0]))
print('Y_output_final_day:', Y_output_dates[-1], weekday(Y_output_dates[-1]))
print('Y_output_days_diff:', daydiff(Y_output_dates[-1], Y_output_dates[0]))

windows = [11, 18, 30, 48, 78, 126, 203, 329]
fib_length = max(windows)

if test_length:
    fib_output_dates = input_df_dates[-fib_length-test_length:-test_length]
else:
    fib_output_dates = input_df_dates[-fib_length:]
fib_input_dates = input_df_dates[-fib_length-test_length-364:-test_length-364]


fib_output_data = input_df[fib_output_dates].values
fib_input_data = input_df[fib_input_dates].values

fib_input_median_list = np.array([np.nanmedian(fib_input_data[:, -w:] , axis=-1) 
                                  for w in windows])
fib_output_median_list = np.array([np.nanmedian(fib_output_data[:, -w:] , axis=-1) 
                                   for w in windows])
    
fib_input_median = np.nan_to_num(np.nanmedian(fib_input_median_list.T, axis=-1))
fib_output_median = np.nan_to_num(np.nanmedian(fib_output_median_list.T, axis=-1))

del fib_output_data
del fib_input_data
del fib_output_median_list
del fib_input_median_list
gc.collect()

Y_input_fib = fib_input_median.reshape(-1,1)
Y_output_fib = fib_output_median.reshape(-1,1)

X_input_raw = input_df[X_input_dates].values
Y_input_raw = input_df[Y_input_dates].values
X_output_raw = input_df[X_output_dates].values
if test_length:
    Y_output_raw = input_df[Y_output_dates].values
    
X_input_num = np.nan_to_num(X_input_raw)
Y_input_num = np.nan_to_num(Y_input_raw)
X_output_num = np.nan_to_num(X_output_raw)
if test_length:
    Y_output_num = np.nan_to_num(Y_output_raw)
    
def log(X):
    return np.log10(X + 1.0)
def unlog(X):
    return np.clip(np.power(10., X) - 1.0, 0.0, None)



X_input_raw_log = log(X_input_raw)  # Contain nan
Y_input_raw_log = log(Y_input_raw)
X_output_raw_log = log(X_output_raw)
if test_length:
    Y_output_raw_log = log(Y_output_raw)



X_input_log = log(X_input_num)  # Do not contain nan
Y_input_log = log(Y_input_num)
X_output_log = log(X_output_num)
if test_length:
    Y_output_log = log(Y_output_num)

output_center = log(fib_output_median).reshape(-1,1) # Fib. median as the center
input_center = log(fib_input_median).reshape(-1,1)


X_input_center = input_center # Use same center for X and Y
Y_input_center = input_center
X_output_center = output_center
Y_output_center = output_center

default_input_scale = np.nanmedian(np.nanstd(X_input_raw_log, axis=-1))  # Do not include nan
default_output_scale = np.nanmedian(np.nanstd(X_output_raw_log, axis=-1))  # Do not include nan
default_scale = np.mean([default_input_scale, default_output_scale])
print('default_scale:', default_scale)

# Treat nan as 0
input_scale = np.std(X_input_log, axis=-1).reshape(-1,1)
output_scale = np.std(X_output_log, axis=-1).reshape(-1,1)

input_scale[input_scale == 0.0] = default_scale  # Prevent divid by zero 
output_scale[output_scale == 0.0] = default_scale  # Prevent divid by zero

X_input_scale = input_scale
Y_input_scale = input_scale
X_output_scale = output_scale
Y_output_scale = output_scale


def transform(data_ori, center, scale):
    return (data_ori - center) / scale
def untransform(data, center, scale):
    return data * scale + center

# Normalize before nan->0
X_input = np.nan_to_num(transform(X_input_raw_log, X_input_center, input_scale))
X_output = np.nan_to_num(transform(X_output_raw_log, X_output_center, output_scale))
Y_input = np.nan_to_num(transform(Y_input_raw_log, Y_input_center, input_scale))
if test_length:
    Y_output = np.nan_to_num(transform(Y_output_raw_log, Y_output_center, output_scale))
    
def check_nan(X):
    return [x for x in X if np.isnan(x).any()]

def group_index(logx):
    if logx < 1.0: return 0
    elif logx < 2.0: return 1
    elif logx < 4.0: return 2
    else: return 3
group_index_v = np.vectorize(group_index)
gp_list = list(range(4))

# Group using Y center (Fib Median)
input_gp = group_index_v(Y_input_center).reshape(-1)
output_gp = group_index_v(Y_output_center).reshape(-1)

# group counts
gp_input_counts = [0] * len(gp_list)
for x in input_gp: gp_input_counts[x] += 1
print('gp_input_counts:', gp_input_counts)

# group counts
gp_output_counts = [0] * len(gp_list)
for x in output_gp: gp_output_counts[x] += 1
print('gp_output_counts:', gp_output_counts)

X_input_ori = X_input_num  # Do not contain nan
Y_input_ori = Y_input_num
X_output_ori = X_output_num
if test_length:
    Y_output_ori = Y_output_num
    
X_input_list = [X_input[input_gp == gp] for gp in gp_list]
Y_input_list = [Y_input[input_gp == gp] for gp in gp_list]
X_output_list = [X_output[output_gp == gp] for gp in gp_list]
if test_length:
    Y_output_list = [Y_output[output_gp == gp] for gp in gp_list]
A_input = np.concatenate((Y_input_center, input_scale), axis=1)
A_output = np.concatenate((Y_output_center, output_scale), axis=1)

A_input_list = [A_input[input_gp == gp] for gp in gp_list]
A_output_list = [A_output[output_gp == gp] for gp in gp_list]

a_length = len(A_input[0])

y_eval_length = 63
y_not_eval = y_length - y_eval_length  # number of days we don't evaluate their scores
print("Number of day we don't evaluate:", y_not_eval)
if test_length and y_not_eval > 0:
    Y_output_raw[:,:y_not_eval] = np.nan
if test_length and y_not_eval > 0:
    print(np.sum(~np.isnan(Y_output_raw[0])), y_eval_length, y_length)
if test_length and y_not_eval > 0:
    print(Y_output_raw[0:2,y_not_eval-1:y_not_eval+1])
def make_shuffle_index(n, seed=None):
    shuffle_index = np.arange(n)
    np.random.seed(seed)  # you can fix the initial seed for comparison purpose
    np.random.shuffle(shuffle_index)
    return shuffle_index

import tensorflow.keras.backend as K

def k_smape(y_true, y_pred):
    '''Symmetric mean absolute percentage error for keras metric'''
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true) + K.abs(y_pred),
                                            K.epsilon(), None))
    return 200. * K.mean(diff, axis=-1)

def smape(y_true, y_pred, axis=None):
    '''Symmetric mean absolute percentage error'''
    diff = np.abs((y_true - y_pred) / 
                  np.clip(np.abs(y_true) + np.abs(y_pred), np.float32(1e-07), None))
    return np.float32(200.) * np.nanmean(diff, axis=axis)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam

n_ens = 5  # Number of ensembles
ens_list = list(range(n_ens))

#load exisiting model
load_previous_models = False  # Switch for determine whether to load the exisiting model or not
if load_previous_models:
    print('Load Previous Models')
    models_ens = [[load_model('../models/' + model_name + '-' + str(run) + '-' + str(gp) +'.h5',
                              custom_objects={'k_smape': k_smape})
                   for gp in gp_list]
                  for run in ens_list]
else: print('Not loading.')

if not load_previous_models:
    models_ens = []
    for run in ens_list:
        print('Run', run, end=': ')
        models = []
        for gp in gp_list:
            print('Group-', gp, sep='', end=' ')
            layer_0 = Input(shape=(x_length,), name='x_input')
            layer_t = Reshape((64, 1))(layer_0)
            layer_t = Conv1D(140, kernel_size=3, activation='relu')(layer_t)
            layer_t = AveragePooling1D(pool_size=2)(layer_t)
            #layer_t=Reshape((140, 1))(layer_t)
            layer_cnn_x = Flatten()(layer_t)

            layer_a = Input(shape=(a_length,), name='a_input')

            layer_t = concatenate([layer_cnn_x, layer_a])

            layer_t = Dense(130, activation='relu')(layer_t)
            layer_t = Dropout(0.25)(layer_t)
            layer_t = Dense(120, activation='relu')(layer_t)
            layer_t = Dropout(0.5)(layer_t)
            layer_f = Dense(y_length)(layer_t)

            model = Model(inputs=[layer_0, layer_a], outputs=layer_f)
            model.compile(optimizer='adam',
                          loss='mean_absolute_error', metrics=[k_smape])
            models.append(model)
        models_ens.append(models)
        print('')
        


print('N of ensemble, N of groups:', np.array(models_ens).shape)


epochs_list = [0,40,40,60]  # Epochs setting for each group

#time
if load_previous_models:
    print('Use Previous Model. Not training.')
else:
    hists_ens = []
    shuffle_indexs_ens = []
    for run, models in zip(ens_list, models_ens):
        print('=== Run:', run+1, '/', len(ens_list), '===')
        hists = []
        shuffle_indexs = []
        for (gp, model, X_in, Y_in, epochs, A_in) in zip(gp_list, models,
                                           X_input_list, Y_input_list, 
                                           epochs_list, A_input_list):
            print('--- Group:', gp+1, '/', len(gp_list), '---')
            
            # Shuffle data
            shuffle_index = make_shuffle_index(len(X_in))
            X_in_sh = X_in[shuffle_index]
            Y_in_sh = Y_in[shuffle_index]
            A_in_sh = A_in[shuffle_index]

            hist = model.fit([X_in_sh, A_in_sh], Y_in_sh, batch_size=128, 
                             epochs=epochs, 
                             validation_split=0.2, verbose=2)
            hists.append(hist)
            shuffle_indexs.append(shuffle_index)
        hists_ens.append(hists)
        shuffle_indexs_ens.append(shuffle_indexs)

Y_input_pred_list_ens = []
for models, run in zip(models_ens, ens_list):
    print('Run', run, end=': ')
    Y_input_pred_list = []
    for (gp, model, X, A) in zip(gp_list, models, X_input_list, A_input_list):
        print(gp, end=' ')
        Y_input_pred_list.append(model.predict([X, A]))
    Y_input_pred_list_ens.append(Y_input_pred_list)
    print('')

input_index_range = np.arange(len(input_gp)); #input_index_range
# list for original index
input_index_list = [input_index_range[input_gp == gp] 
                     for gp in gp_list]

input_index_list_comb = np.concatenate(input_index_list); #input_index_list_comb
Y_input_pred_ens = []
for Y_input_pred_list in Y_input_pred_list_ens:
    Y_input_pred_comb = np.concatenate(Y_input_pred_list)
    Y_input_pred = [0]*len(input_index_list_comb)

    for index, y in zip(input_index_list_comb, Y_input_pred_comb):
        Y_input_pred[index] = y

    Y_input_pred = np.array(Y_input_pred)  # make it an numpy array (which will also make a copy)
    Y_input_pred_ens.append(Y_input_pred)

del Y_input_pred_comb
# Inverse transform Y_input_pred to original Y
Y_input_pred_ori_ens = []
for Y_input_pred in Y_input_pred_ens:
    Y_input_pred_ori = unlog(untransform(Y_input_pred, Y_input_center, Y_input_scale))
    Y_input_pred_ori_ens.append(Y_input_pred_ori)

Y_input_pred_ori_ens = np.array(Y_input_pred_ori_ens)


