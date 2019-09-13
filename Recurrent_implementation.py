

import numpy as np
import tensorflow as tf
from tensorflow import keras
#from plot import plot_loss
#from time_history import TimeHistory, total_time
from matplotlib import pyplot as plt
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
input_df_np_nums=(input_df.values[:,1:]).astype(float)

def weekday(datestr):
    return datetime.strptime(datestr,'%Y-%m-%d').weekday()
def daydiff(dstr1,dstr2):
    return datetime.strptime(dstr1,'%Y-%m-%d') - datetime.strptime(dstr2,'%Y-%m-%d')
tweny_first=input_df_np_nums[:200,:]
normal_tweny_first=input_df_np_nums[:200,:]
for i in range(200):
    normal_tweny_first[i,:]=(tweny_first[i,:]-np.nanmedian(tweny_first[i,:]))/np.nanstd(tweny_first[i,:])

normal_tweny_first=np.nan_to_num(normal_tweny_first)

del input_df
training=normal_tweny_first[:,:600]


testing=normal_tweny_first[:,700:800]

validation=normal_tweny_first[:,600:700]

train = keras.preprocessing.sequence.TimeseriesGenerator(np.transpose(training), np.transpose(training),
                               length=5, sampling_rate=1,stride =1,
                               batch_size=1)
valid = keras.preprocessing.sequence.TimeseriesGenerator(np.transpose(validation), np.transpose(validation),
                               length=5, sampling_rate=1,stride =1,
                               batch_size=1)

test = keras.preprocessing.sequence.TimeseriesGenerator(np.transpose(testing), np.transpose(testing),
                               length=5, sampling_rate=1,stride =1,
                               batch_size=1)

RNN1w = keras.models.Sequential()
RNN1w.add(keras.layers.SimpleRNN(8, input_shape=(5,200)))

RNN1w.add(keras.layers.Flatten())
RNN1w.add(keras.layers.Dense(200))
RNN1w.add(keras.layers.Dense(100))
RNN1w.add(keras.layers.Dense(50))
RNN1w.add(keras.layers.Dense(100))
RNN1w.add(keras.layers.Dense(200))
RNN1w.compile(loss='mae', optimizer=keras.optimizers.RMSprop())
RNN1w_history = RNN1w.fit_generator(train, epochs=80, validation_data=valid)
prediction = RNN1w.predict_generator(test)
pr_y = np.zeros((len(prediction),200))
t_y = np.zeros((len(test),200))
for j in range (len(prediction)):
    for num in range(200):
            pr_y[j,num]=prediction[j][num]
            t_y[j,num]=(test[j][1])[0][num]

    
predictes_loss=np.abs(pr_y)-np.abs(t_y)
pr_y_sum=np.zeros((1,len(prediction)))
t_y_sum=np.zeros((1,len(test)))
for i in range(len(prediction)):
    pr_y_sum[:,i]=np.sum(np.abs(prediction[i,:]-(test[j][1])[0,:]))
    t_y_sum[:,i]=np.sum((test[j][1])[0,:])
plt.plot(t_y_sum, pr_y_sum, 'ro')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.show()