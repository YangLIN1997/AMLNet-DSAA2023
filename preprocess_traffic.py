from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import h5py
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--L', default=7, type=int,help='Number of input period')
parser.add_argument('--H', default=1, type=int,help='Number of forecasting period')



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
    if data2 is not None:
        time_len = data2.shape[0]
        total_windows = n_id * (time_len - input_size) // stride_size
        print("windows pre2: ", total_windows, "   No of days:", total_windows / n_id)
        # if train: windows_per_series -= (stride_size-1) // stride_size
        x_input2 = np.zeros((total_windows, window_size, num_covariates), dtype='float32')
        label2 = np.zeros((total_windows, window_size), dtype='float32')
        for i in range(total_windows // n_id):
            window_start = stride_size * i
            window_end = window_start + window_size
            x_input2[i * n_id:(i + 1) * n_id, 0, 0] = (x_input2[i * n_id:(i + 1) * n_id, 0, 0] - data_mean) / data_scale
            x_input2[i * n_id:(i + 1) * n_id, 1:, 0] = data2[window_start:window_end - 1, :, 0].swapaxes(0, 1).reshape(
                -1, window_size - 1)
            x_input2[i * n_id:(i + 1) * n_id, :, 1:] = data2[window_start:window_end, :, 1:].swapaxes(0, 1).reshape(-1,
                                                                                                                    window_size,
                                                                                                                    num_covariates - 1)
            label2[i * n_id:(i + 1) * n_id, :] = data2[window_start:window_end, :, 0].swapaxes(0, 1).reshape(-1,
                                                                                                             window_size)
        zeros_index = np.zeros(x_input2.shape[0])
        for i in range(window_size // stride_size):
            var = np.var(x_input2[:, i * stride_size + 1:(i + 1) * stride_size, 0] * data_scale[
                x_input2[:, 0, -1].astype(np.int)].reshape(-1, 1) + data_mean[
                             x_input2[:, 0, -1].astype(np.int)].reshape(-1, 1), axis=1)
            zeros_index += (var < 1e-3)
        zeros_index = np.where((zeros_index > 0))[0]
        x_input2 = np.delete(x_input2, zeros_index, axis=0)
        label2 = np.delete(label2, zeros_index, axis=0)

        x_input = np.concatenate((x_input, x_input2), axis=0)
        label = np.concatenate((label, label2), axis=0)

    prefix = os.path.join(save_path, name + '_')
    np.save(prefix + 'data_' + task + save_name, x_input)
    print(prefix + 'data_' + task + save_name, x_input.shape)
    # print(label[0,:24])
    # print(label[0,:24]*data_scale[0]+data_mean[0])
    # if name == 'test':
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
    # Standardlize data
    data_scale = np.zeros(n_id)
    data_mean = np.zeros(n_id)
    for i in range(n_id):
        st_scaler = StandardScaler()
        # st_scaler.fit(train_data[data_start[i]:, i, :-1])
        # train_data[:, i, :-1] = st_scaler.transform(train_data[:, i, :-1])
        # valid_data[:, i, :-1] = st_scaler.transform(valid_data[:, i, :-1])
        # test_data[:, i, :-1] = st_scaler.transform(test_data[:, i, :-1])        
        st_scaler.fit(train_data[:, i, 0].reshape(-1,1))
        train_data[:, i, 0] = st_scaler.transform(train_data[:, i, 0].reshape(-1,1)).reshape(-1,)
        valid_data[:, i, 0] = st_scaler.transform(valid_data[:, i, 0].reshape(-1,1)).reshape(-1,)
        test_data[:, i, 0] = st_scaler.transform(test_data[:, i, 0].reshape(-1,1)).reshape(-1,)
        data_scale[i] = st_scaler.scale_[0]
        data_mean[i] = st_scaler.mean_[0]
    # Prepare data
    # prep_data(train_data[0:480], data_mean, data_scale, task, name='train', data2 = train_data[0:480])
    prep_data(train_data[:train_data1.shape[0]], data_mean, data_scale, task, name='train', data2 = None)
    # prep_data(train_data[:train_data1.shape[0]], data_mean, data_scale, task, name='train', data2=None)
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
        covariates[i, :, 1] = input_time.weekday()
        covariates[i, :, 2] = input_time.hour
        covariates[i, :, 3] = input_time.month
    for i in range(n_id):
        covariates[:, i, -1] = i
        cov_age = np.zeros((times.shape[0],))
        # cov_age[:] = stats.zscore(np.arange(times.shape[0] ))
        cov_age = np.arange(times.shape[0])
        covariates[:, i, 4] = cov_age
    # print(np.max(covariates[:,:,1]))
    # print(np.max(covariates[:,:,2]))
    # print(np.max(covariates[:,:,3]))
    # print(np.max(covariates[:,:,4]))
    # print(np.max(covariates[:,:,5]))
    # for i in range(1,num_covariates-1):
    # covariates[:,:,i] = stats.zscore(covariates[:,:,i])
    return covariates


if __name__ == '__main__':

    global save_path
    save_name = 'traffic'
    zip_name = 'PEMS-SF.zip'
    day_steps = 24
    args = parser.parse_args()
    window_size = day_steps*(args.L+args.H)
    stride_size = day_steps*args.H
    pred_days = args.L
    given_days = args.H
    num_covariates = 6 # z;time feature;id
    
    save_path = os.path.join('../data', save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with ZipFile(os.path.join(save_path, zip_name)) as zfile:
        zfile.extractall(save_path)
    # The PEMS_train textfile has 263 lines. Each line describes a time-series provided as a matrix.
    # The matrix syntax is that of Matlab, e.g. [ a b ; c d] is the matrix with row vectors [a b] and [c d] in that order.
    # Each matrix describes the different occupancies rates (963 lines, one for each station/detector)
    # sampled every 10 minutes during the day (144 columns)
    file_name = "PEMS_train"
    file_path = os.path.join(save_path, file_name)
    print("file_path: ",file_path)
    mynumbers = []
    with open(file_path) as f:
        for line in f:
            mynumbers.append([n.split(" ") for n in line[1:-2].split(';')])
    PEMS_train = np.array(mynumbers).astype(np.float)
    #The PEMS_trainlabel text describes, for each day of measurements described above
    file_name = "PEMS_test"
    file_path = os.path.join(save_path, file_name)
    mynumbers = []
    with open(file_path) as f:
        for line in f:
            mynumbers.append([n.split(" ") for n in line[1:-2].split(';')])
    PEMS_test = np.array(mynumbers).astype(np.float)
    # The permutation that I used to shuffle the dataset is given in the randperm file.
    # If you need to rearrange the data so that it follows the calendar order,
    # you should merge train and test samples and reorder them using the inverse permutation of randperm.
    file_name = "randperm"
    file_path = os.path.join(save_path, file_name)
    f = open(file_path, "r")
    randperm = f.read()
    randperm = np.array(randperm[1:-2].split(" ")).astype(np.int) - 1
    idx = np.empty_like(randperm)
    idx[randperm] = np.arange(len(randperm))
    PEMS = np.concatenate((PEMS_train, PEMS_test), axis=0)
    PEMS = PEMS[idx]
    n_id = PEMS.shape[1]
    # remove the holiday based on the description of the data set
    dateList = pd.date_range(start='01/01/2008', end='30/03/2009')
    dateRemove = pd.to_datetime(
        ['2008-01-01', '2008-01-21', '2008-02-18', '2008-03-31', '2008-03-09', '2008-05-26', '2008-07-04', '2008-09-01',
         '2008-11-11', '2008-11-17', '2008-12-25', '2009-01-01', '2009-01-21', '2009-02-16', "2009-03-08"])
    dateList = [s for s in dateList if s not in dateRemove]
    hour_list = []
    for nDate in dateList:
        for nHour in range(24):
            tmp_timestamp = nDate + datetime.timedelta(hours=nHour)
            hour_list.append(tmp_timestamp)
    hour_list = np.array(hour_list)
    date_time = pd.to_datetime(hour_list)
    data_frame = pd.DataFrame(PEMS.swapaxes(1, 2).reshape(-1, n_id))
    LocalTime = [datetime.datetime(2008, 1, 1) + datetime.timedelta(minutes=i * 10) for i in range(data_frame.shape[0])]
    data_frame = data_frame.set_index(pd.to_datetime(LocalTime))
    data_frame = data_frame.resample('1H', label='left', closed='left').sum()
    data_frame = data_frame.set_index(pd.to_datetime(hour_list))
    # set hoildays as zeros
    hour_list = []
    for nDate in dateRemove:
        for nHour in range(24):
            tmp_timestamp = nDate + datetime.timedelta(hours=nHour)
            hour_list.append(tmp_timestamp)
    hour_list = np.array(hour_list)
    dateRemove_time = pd.to_datetime(hour_list)
    data_frame_Remove = pd.DataFrame(np.zeros((dateRemove_time.shape[0], data_frame.shape[1])))
    data_frame_Remove = data_frame_Remove.set_index(dateRemove_time)
    data_frame = pd.concat((data_frame, data_frame_Remove)).sort_index()
    print('From: ', data_frame.index[0], 'to: ', data_frame.index[-1])
    # data_frame.fillna(0, inplace=True)

    # visualize(data_frame, 365+22, day_num=20, save_name=save_name)
    n_id = data_frame.shape[1]
    n_day = data_frame.shape[0]/stride_size
    print('total days:', n_day)
    print('total samples:', data_frame.shape[0])
    print('total series:', data_frame.shape[1])

    # # For gridsearch
    # train_start = '2008-01-01 00:00:00'
    # train_end = '2008-06-08 23:00:00'
    # valid_start = '2008-06-01 00:00:00'  # need additional 7 days as given info
    # valid_end = '2008-06-14 23:00:00'
    # test_start = '2008-06-08 00:00:00' #need additional 7 days as given info
    # test_end = '2008-06-21 23:00:00'
    # train_start2 = '2008-06-22 00:00:00'
    # train_end2 = '2009-03-30 23:00:00'
    # prepare(task='search_')

    # # For inference
    # train_start = '2008-01-01 00:00:00'
    # train_end = '2008-06-14 23:00:00'
    # valid_start = '2008-06-08 00:00:00'  # need additional 7 days as given info
    # valid_end = '2008-06-21 23:00:00'
    # test_start = '2008-06-15 00:00:00' #need additional 7 days as given info
    # test_end = '2008-06-28 23:00:00'
    # train_start2 = '2008-06-22 00:00:00'
    # train_end2 = '2009-03-30 23:00:00'
    # prepare(task='')

    # For inference
    train_start = '2008-01-01 00:00:00'
    train_end = '2008-08-05 23:00:00'
    # train_end = '2006-08-24 23:00:00'
    valid_start = '2008-08-05 00:00:00' #need additional 7 days as given info
    valid_end = '2008-08-18 23:00:00'
    test_start = '2008-08-18 00:00:00' #need additional 7 days as given info
    test_end = '2008-08-31 23:00:00'
    prepare(task='')
