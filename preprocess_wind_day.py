from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import h5py
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from zipfile import ZipFile
import datetime

def prep_data(data, data_mean, data_scale, task='search_', name='train', data2=None):
    input_size = window_size - stride_size
    time_len = data.shape[0]
    total_windows = n_id * ((time_len - input_size) // stride_size)
    print("windows pre: ", total_windows, "   No of days:", total_windows / n_id)
    # if train: windows_per_series -= (stride_size-1) // stride_size
    x_input = np.zeros((total_windows, window_size, num_covariates), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')
    for i in range(total_windows // n_id):
        window_start = stride_size * i
        window_end = window_start + window_size
        x_input[i * n_id:(i + 1) * n_id, 0, 0] = (x_input[i * n_id:(i + 1) * n_id, 0, 0] - data_mean) / data_scale
        x_input[i * n_id:(i + 1) * n_id, 1:, 0] = data[window_start:window_end - 1, :, 0].swapaxes(0, 1).reshape(-1,
                                                                                                                 window_size - 1)
        x_input[i * n_id:(i + 1) * n_id, :, 1:] = data[window_start:window_end, :, 1:].swapaxes(0, 1).reshape(-1,
                                                                                                              window_size,
                                                                                                              num_covariates - 1)
        label[i * n_id:(i + 1) * n_id, :] = data[window_start:window_end, :, 0].swapaxes(0, 1).reshape(-1, window_size)
    zeros_index = np.zeros(x_input.shape[0])
    for i in range(window_size // stride_size):
        var = np.var(x_input[:, i * stride_size + 1:(i + 1) * stride_size, 0] * data_scale[
            x_input[:, 0, -1].astype(np.int)].reshape(-1, 1) + data_mean[x_input[:, 0, -1].astype(np.int)].reshape(-1,
                                                                                                                   1),
                     axis=1)
        # print(x_input)
        zeros_index += (var < 1e-3)
    zeros_index = np.where((zeros_index > 0))[0]
    x_input = np.delete(x_input, zeros_index, axis=0)
    label = np.delete(label, zeros_index, axis=0)
    print('x_input', x_input.shape)
    # x = np.arange(window_size)
    # if zeros_index.shape[0]>0:
    #     f = plt.figure()
    #     plt.plot(x, x_input[zeros_index[0].astype(np.int), :, 0] * data_scale[x_input[zeros_index[0].astype(np.int), :, -1].astype(np.int) ] + data_mean[x_input[zeros_index[0].astype(np.int), :, -1].astype(np.int) ], color='b')
    #     f.savefig(name+save_name + '_visual_removed.png')
    #     plt.close()

    prefix = os.path.join(save_path, name + '_')
    np.save(prefix + 'data_' + task + save_name, x_input)
    print(prefix + 'data_' + task + save_name, x_input.shape)
    # print(label[0,:24])
    # print(label[0,:24]*data_scale[0]+data_mean[0])
    if name == 'test':
        # np.save(prefix+'mean_'+task+save_name, data_mean)
        # np.save(prefix+'scale_'+task+save_name, data_scale)
        np.save(prefix + 'mean_' + task + save_name, data_mean[x_input[:, 0, -1].astype(np.int)])
        np.save(prefix + 'scale_' + task + save_name, data_scale[x_input[:, 0, -1].astype(np.int)])
    np.save(prefix + 'label_' + task + save_name, label)


def prepare(task='search_'):
    covariates = gen_covariates(data_frame[train_start:test_end].index, num_covariates, n_id)
    train_data1 = covariates[:data_frame[:train_end].shape[0]].copy()
    valid_data = covariates[data_frame[:valid_start].shape[0] - 1:data_frame[:valid_end].shape[0]].copy()
    test_data = covariates[data_frame[:test_start].shape[0] - 1:data_frame[:test_end].shape[0]].copy()
    # train_data2 = covariates[data_frame[:train_start2].shape[0] - 1:data_frame[:train_end2].shape[0]].copy()
    # print(train_data1.shape, valid_data.shape, test_data.shape)
    # valid_data = gen_covariates(data_frame[valid_start:valid_end].index, num_covariates, n_id)
    valid_data[:, :, 0] = data_frame[valid_start:valid_end].copy()
    # test_data = gen_covariates(data_frame[test_start:test_end].index, num_covariates, n_id)
    test_data[:, :, 0] = data_frame[test_start:test_end].copy()
    # train_data1 = gen_covariates(data_frame[train_start:train_end].index, num_covariates, n_id)
    train_data1[:, :, 0] = data_frame[train_start:train_end].copy()
    # train_data2 = gen_covariates(data_frame[train_start2:train_end2].index, num_covariates, n_id)
    # train_data2[:, :, 0] = data_frame[train_start2:train_end2]
    # train_data = np.concatenate((train_data1,train_data2),axis=0)
    train_data = train_data1

    # print('test: ',test_data[0:24,0,0])

    # Standardlize data
    # temp = train_data[data_start[0]:, 0, :]
    # for i in range(1,n_id):
    #     temp = np.concatenate((temp,train_data[data_start[i]:, i, :]),axis=0)
    # temp = temp.reshape((-1,num_covariates))
    # train_data = train_data.reshape((-1,num_covariates))
    # valid_data = valid_data.reshape((-1,num_covariates))
    # test_data = test_data.reshape((-1,num_covariates))
    # # print(np.mean(data_frame.values))
    # # print(np.mean(valid_data,axis=0)[0])
    # # print(np.mean(train_data,axis=0)[0])
    # # print(np.max(data_frame.values))
    # st_scaler = StandardScaler()
    # st_scaler.fit(temp[:,0:-1])
    # # st_scaler.fit(train_data[:,0:-1])
    # train_data[:,0:-1] = st_scaler.transform(train_data[:,0:-1])
    # valid_data[:,0:-1] = st_scaler.transform(valid_data[:,0:-1])
    # test_data[:,0:-1] = st_scaler.transform(test_data[:,0:-1])
    # # st_scaler.fit(train_data[:,0].reshape(-1, 1))
    # # train_data[:,0] = st_scaler.transform(train_data[:,0].reshape(-1, 1)).reshape(-1,)
    # # valid_data[:,0] = st_scaler.transform(valid_data[:,0].reshape(-1, 1)).reshape(-1,)
    # # test_data[:,0] = st_scaler.transform(test_data[:,0].reshape(-1, 1)).reshape(-1,)
    # data_scale = st_scaler.scale_
    # data_mean = st_scaler.mean_
    # # print(data_scale[0],data_mean[0])
    # # print(np.mean(train_data,axis=0)[0])
    # # print(np.mean(valid_data,axis=0)[0])
    # # print(np.mean(test_data,axis=0)[0])

    # train_data = train_data.reshape((-1,n_id,num_covariates))
    # valid_data = valid_data.reshape((-1,n_id,num_covariates))
    # test_data = test_data.reshape((-1,n_id,num_covariates))
    # print(np.mean(train_data[:,:,0]))
    # Standardlize data
    data_scale = np.zeros(n_id)
    data_mean = np.zeros(n_id)

    for i in range(n_id):
        st_scaler = StandardScaler()
        # st_scaler.fit(train_data[data_start[i]:, i, :-1])
        # train_data[:, i, :-1] = st_scaler.transform(train_data[:, i, :-1])
        # valid_data[:, i, :-1] = st_scaler.transform(valid_data[:, i, :-1])
        # test_data[:, i, :-1] = st_scaler.transform(test_data[:, i, :-1])        
        st_scaler.fit(train_data[data_start[i]:, i, 0].reshape(-1,1))
        train_data[:, i, 0] = st_scaler.transform(train_data[:, i, 0].reshape(-1,1)).reshape(-1,)
        valid_data[:, i, 0] = st_scaler.transform(valid_data[:, i, 0].reshape(-1,1)).reshape(-1,)
        test_data[:, i, 0] = st_scaler.transform(test_data[:, i, 0].reshape(-1,1)).reshape(-1,)
        data_scale[i] = st_scaler.scale_[0]
        data_mean[i] = st_scaler.mean_[0]
    # Prepare data
    # prep_data(train_data[0:480], data_mean, data_scale, task, name='train', data2 = train_data[0:480])
    # prep_data(train_data[:train_data1.shape[0]], data_mean, data_scale, task, name='train', data2 = train_data[train_data1.shape[0]:])
    prep_data(train_data, data_mean, data_scale, task, name='train', data2=None)
    prep_data(valid_data, data_mean, data_scale, task, name='valid', data2=None)
    prep_data(test_data, data_mean, data_scale, task, name='test', data2=None)


def visualize(data, day_start, day_num, save_name):
    x = np.arange(stride_size * day_num)
    f = plt.figure()
    plt.plot(x, data[day_start * stride_size:day_start * stride_size + stride_size * day_num].values[:, 4], color='b')
    f.savefig('visual_' + save_name + '.png')
    plt.close()


def gen_covariates(times, num_covariates, n_id):
    covariates = np.zeros((times.shape[0], n_id, num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, :, 1] = input_time.day
        covariates[i, :, 2] = input_time.month
    for i in range(n_id):
        covariates[:, i, -1] = i
        cov_age = np.zeros((times.shape[0],))
        cov_age[data_start[i]:] = stats.zscore(np.arange(times.shape[0] - data_start[i]))
        covariates[:, i, 3] = cov_age
    for i in range(1,num_covariates-1):
        covariates[:,:,i] = stats.zscore(covariates[:,:,i])
    return covariates


if __name__ == '__main__':

    global save_path
    save_name = 'wind'
    zip_name = 'EMHIRESPV_TSh_CF_Country_19862015.zip'
    window_size = 30*2
    stride_size = 1
    input_size = 30
    pred_days = 30
    given_days = 30
    num_covariates = 5

    save_path = os.path.join('data', save_name)
    data_path = os.path.join(save_path, 'EMHIRESPV_TSh_CF_Country_19862015.csv')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with ZipFile(os.path.join(save_path, zip_name)) as zfile:
        zfile.extractall(save_path)
    data_frame = pd.read_csv(data_path)
    LocalTime = [datetime.datetime(1986, 1, 1) + datetime.timedelta(hours=i) for i in range(data_frame.shape[0])]
    data_frame = data_frame.set_index(pd.to_datetime(LocalTime))
    # input_size = window_size-stride_size
    data_frame = data_frame.resample('1D', label='left', closed='left').sum()

    data_frame = data_frame.drop(columns=['CY'])

    print('From: ',data_frame.index[0],'to: ',data_frame.index[-1])
    visualize(data_frame, 50, day_num=30, save_name = save_name)
    
    n_id = data_frame.shape[1]
    n_day = data_frame.shape[0]/stride_size
    print('total 30 days:', n_day)
    print('total samples:', data_frame.shape[0])
    print('total series:', data_frame.shape[1])
    data_start = (data_frame.values!=0).argmax(axis=0) #find first nonzero value in each time series
    data_start = (data_start // stride_size) * stride_size

    # # For gridsearch
    train_start = '1986-01-01 00:00:00'
    train_end = '2014-04-10 00:00:00'
    valid_start = '2014-04-10 00:00:00' #need additional 30 days as given info
    valid_end = '2014-11-06 00:00:00'
    test_start = '2014-11-06 00:00:00' #need additional 30 days as given info
    test_end = '2015-06-04 00:00:00'
    prepare(task='search_')

    # For inference
    train_start = '1986-01-01 00:00:00'
    train_end = '2014-11-06 00:00:00'
    valid_start = '2014-10-09 00:00:00' #need additional 30 days as given info
    valid_end = '2015-06-05 00:00:00'
    test_start = '2015-05-06 00:00:00' #need additional 30 days as given info
    test_end = '2015-12-31 00:00:00'
    prepare(task='')

    # # # For gridsearch
    # train_start = '1986-01-01 00:00:00'
    # train_end = '2014-04-10 23:00:00'
    # valid_start = '2014-04-10 00:00:00' #need additional 7 days as given info
    # valid_end = '2014-11-06 23:00:00'
    # test_start = '2014-11-06 00:00:00' #need additional 7 days as given info
    # test_end = '2015-06-04 23:00:00'
    # prepare(task='search_')

    # # For inference
    # train_start = '1986-01-01 00:00:00'
    # train_end = '2014-11-06 23:00:00'
    # valid_start = '2014-11-06 00:00:00' #need additional 7 days as given info
    # valid_end = '2015-06-04 23:00:00'
    # test_start = '2015-06-04 00:00:00' #need additional 7 days as given info
    # test_end = '2015-12-31 23:00:00'
    # prepare(task='')