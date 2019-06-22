import pandas as pd
import numpy as np
import pickle
import scipy.io as sio
import h5py
import time
import os
from scipy.ndimage import convolve, convolve1d
from util.preprocessing import *
from util.params import *


def load_data(filename, split):
    if len(filename)==2:
        d1 = sio.loadmat(filename[0])['p_map']
        d2 = sio.loadmat(filename[1])['d_map']
        data = np.concatenate((d1[:,:,:,np.newaxis], d2[:,:,:,np.newaxis]), axis=3)
    train = data[0:split[0],:,:,:]
    validate = data[split[0]:split[0]+split[1],:,:,:]
    test = data[split[0]+split[1]:split[0]+split[1]+split[2],:,:,:]
    return data, train, validate, test

def load_npy_data(filename, split):
    if len(filename)==2:
        d1 = np.load(filename[0])
        d2 = np.load(filename[1])
        data = np.concatenate((d1[:,:,:,np.newaxis], d2[:,:,:,np.newaxis]), axis=3)
    # train = data[0:split[0],:,:,:]
    # validate = data[split[0]:split[0]+split[1],:,:,:]
    # test = data[split[0]+split[1]:split[0]+split[1]+split[2],:,:,:]
    train = data[:split['val'], :, :, :]
    val = data[split['val']:split['test'], :, :, :]
    test = data[split['test']:, :, :, :]
    return data, train, val, test

def load_BJdata(fpath, split, T=24):
    arr_data = []
    arr_tstamps = []
    for year in range(13, 17):
        fname = os.path.join(fpath, 'BJ{}_M32x32_T30_InOut.h5'.format(year))
        f = h5py.File(fname, 'r')
        if T == 24: # 48 tslots one day
            data = f['data'].value.transpose([1,0,2,3])
            nflow, nslots, m, n  = data.shape
            data = data.reshape([nflow, int(nslots/2), 2, m, n])
            data = data.sum(axis=2)
            data = data.transpose([1,2,3,0])
            timestamps = f['date'].value.reshape([int(nslots/48), 48])[:,:24].flatten()
            arr_tstamps.append(timestamps)
            arr_data.append(data)
        else: # 48 tslots one day
            data = f['data'].value.transpose([0,2,3,1])
            arr_data.append(data)
            arr_tstamps.append(f['date'].value)
        f.close()
    data = np.concatenate(arr_data)
    all_timestamps = np.concatenate(arr_tstamps)
    all_timestamps = [x.decode('utf-8') for x in all_timestamps]

    t_list = [
        '2013-07-01', '2013-10-29',
        '2014-03-01', '2014-06-27',
        '2015-03-01', '2015-06-30',
        '2015-11-01', '2016-04-09'
    ]
    each_len = []
    t_be = []
    each_len.append(((pd.to_datetime('2013-10-29') - pd.to_datetime('2013-07-01')).days + 1) * T)
    each_len.append(((pd.to_datetime('2014-06-27') - pd.to_datetime('2014-03-01')).days + 1) * T)
    each_len.append(((pd.to_datetime('2015-06-30') - pd.to_datetime('2015-03-01')).days + 1) * T)
    each_len.append(((pd.to_datetime('2016-04-09') - pd.to_datetime('2015-11-01')).days + 1) * T)
    for i in range(4):
        tb = - (pd.to_datetime(t_list[2*i]).isoweekday() - 7) * T
        te = (- pd.to_datetime(t_list[2*i + 1]).isoweekday() - 1) * T
        tb = sum(each_len[:i]) + tb
        te = sum(each_len[:i + 1]) + te
        t_be.append(range(tb, te))

    del_label = set(range(len(data)))
    keep_label = set()
    for item in t_be:
        keep_label = keep_label.union(set(item))
    del_label = list(del_label - keep_label)
    data = np.delete(data, del_label, axis=0)
    all_timestamps = list(np.delete(np.array(all_timestamps), del_label))
    train = data[:split['val'], :, :, :]
    val = data[split['val']:split['test'], :, :, :]
    test = data[split['test']:, :, :, :]

    return data, train, val, test, all_timestamps

def batch_data(data, batch_size=32, input_steps=10, output_steps=10):
    # data: [num, row, col, channel]
    num = data.shape[0]
    # x: [batches, batch_size, input_steps, row, col, channel]
    # y: [batches, batch_size, output_steps, row, col, channel]
    x = []
    y = []
    i = 0
    while i < num-input_steps-output_steps:
        batch_x = []
        batch_y = []
        cur_batch = min(batch_size, num-i-input_steps-output_steps+1)
        for s in range(cur_batch):
            batch_x.append(data[i+s:i+s+input_steps])
            batch_y.append(data[i+s+input_steps:i+s+input_steps+output_steps])
        x.append(batch_x)
        y.append(batch_y)
        i += batch_size
    return x, y

def prepare_all(train_data, val_data, test_data, pre_process):
    if pre_process != None:
        train_data = pre_process.transform(train_data)
    train_x, train_y = batch_data(
        data=train_data, batch_size=FLAGS.batch_size,
        input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps
    )
    if pre_process != None:
        val_data = pre_process.transform(val_data)
    val_x, val_y = batch_data(
        data=val_data, batch_size=FLAGS.batch_size,
        input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps
    )
    if pre_process != None:
        test_data = pre_process.transform(test_data)
    test_x, test_y = batch_data(
        data=test_data, batch_size=FLAGS.batch_size,
        input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps)
    train = {'x': train_x, 'y': train_y}
    val = {'x': val_x, 'y': val_y}
    test = {'x': test_x, 'y': test_y}
    return train, val, test


def batch_data_cpt_ext(data, timestamps, batch_size=32, close=3, period=4, trend=4, T=24, ext_t=[0,0]):
    # data: [num, row, col, channel]
    num = data.shape[0]
    #flow = data.shape[1]
    # x: [batches,
    #[
    #[batch_size, row, col, close*flow],
    #[batch_size, row, col, period*flow],
    #[batch_size, row, col, trend*flow],
    #[batch_size, external_dim]
    #]
    #]
    c = 1
    p = T
    t = T*7
    depends = [ [c*j for j in range(1, close+1)],
                [p*j for j in range(1, period+1)],
                [t*j for j in range(1, trend+1)] ]
    depends = np.asarray(depends)
    i = max(c*close, p*period, t*trend)
    # external feature
    if ext_t[1]-ext_t[0] != len(timestamps):
        ext = external_feature2(timestamps)
        ext = ext[ext_t[0]:ext_t[1],:]
    else:
        ext = external_feature(timestamps)

    # ext plus c p t
    # x: [batches, 4, batch_size]
    # y: [batches, batch_size]
    x = []
    y = []
    while i<num:
        x_b = np.empty(len(depends)+1, dtype=object)
        for d in range(len(depends)):
            x_ = []
            for b in range(batch_size):
                if i+b >= num:
                    break
                x_.append(np.transpose(np.vstack(np.transpose(data[i+b-np.array(depends[d]), :, :, :],[0,3,1,2])), [1,2,0]))
            x_ = np.array(x_)
            x_b[d] = x_
            #x_b.append(x_)
        # external features
        x_b[-1] = ext[i:min(i+batch_size, num)]
        # y
        y_b = data[i:min(i+batch_size, num), :, :, :]
        x.append(x_b)
        #print(y_b.shape)
        y.append(y_b)
        i += batch_size
    return x, y

def external_feature(timestamps):
    vec = [time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps]
    ext = []
    for j in vec:
        v = [0 for _ in range(7)]
        v[j] = 1
        if j >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ext.append(v)
    ext = np.asarray(ext)
    return ext

def external_feature2(timestamps):
    vec_y = np.array([time.strptime(t[:8], '%Y%m%d').tm_year for t in timestamps])
    vec_m = np.array([time.strptime(t[:8], '%Y%m%d').tm_mon for t in timestamps])
    vec_d = np.array([time.strptime(t[:8], '%Y%m%d').tm_mday for t in timestamps])
    vec_h = np.array([int(t[8:10]) for t in timestamps])
    vec_w = np.array([time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps])
    vec_yd = np.array([time.strptime(t[:8], '%Y%m%d').tm_yday for t in timestamps])
    vec_hd = np.array([i for i,t in enumerate(timestamps)])
    vec_z = np.array([1 if x>=5 else 0 for x in vec_w])
    ext = np.vstack([vec_y, vec_m, vec_d, vec_h, vec_w, vec_yd, vec_z, vec_hd])
    ext = np.transpose(ext)

    # normalize
    a = ext - ext.min(axis=0)
    b = ext.max(axis=0) - ext.min(axis=0)
    ext = a / b
    return ext


def kde_data2(dataset='citybike'):
    if dataset in ['citybike', 'nyctaxi']:
        data, train_data, val_data, test_data = \
            load_npy_data(filename=['data/'+dataset+'/p_map.npy', 'data/'+dataset+'/d_map.npy'], split=split_point)
    else:
        # 2013.07.01-2013.10.29, 2014.03.01-2014.06.27, 2015.03.01-2015.06.30, 2015.11.01-2016.04.09
        data, train_data, val_data, test_data, all_timestamps = \
            load_BJdata(fpath='../PCRN/data/TaxiBJ-filt/', split=split_point, T=params['ts_oneday'])

    # select data
    tshape = data.shape
    week_p = tshape[0] // week_len
    week_s = tshape[0] % week_len
    use_len = week_p * (week_len)

    week_data = []
    for i in range(4):
        tmp = data[-use_len+week_len*(4-i-1):-week_len*(i+1)]
        week_data.append(tmp)
    kde_result = np.sum(
        [week_data[0] * 0.4,
         week_data[1] * 0.3,
         week_data[2] * 0.2,
         week_data[3] * 0.1,
         ],
        axis=0
    )
    if week_s == 0:
        kde_result = np.vstack(
            [kde_result[:tshape[0] - kde_result.shape[0]], kde_result])
    else:
        kde_result = np.vstack(
            [kde_result[week_len - week_s:tshape[0] - kde_result.shape[0] + week_len - week_s], kde_result])

    filt = week_data[0].transpose([3, 0, 1, 2]).sum(axis=1)
    filt = np.where(filt < 1e3, 0, 1).transpose([1, 2, 0])

    kde_result = kde_result.transpose([3,1,2,0])
    kde_result = kde_result.reshape([-1,kde_result.shape[-1]])
    kde_result = np.array(
                 list(map(lambda y: convolve(y, [0.33,0.33,0.33], mode='nearest'), kde_result))
                 )
    kde_result = kde_result.reshape([tshape[-1], tshape[1], tshape[2], kde_result.shape[-1]])
    kde_result = kde_result.transpose([3,1,2,0])

    return np.multiply(kde_result, filt)


def kde_data1(dataset='citybike'):
    if dataset in ['citybike', 'nyctaxi']:
        data, train_data, val_data, test_data = \
            load_npy_data(filename=['data/'+dataset+'/p_map.npy', 'data/'+dataset+'/d_map.npy'], split=split_point)
    else:
        # 2013.07.01-2013.10.29, 2014.03.01-2014.06.27, 2015.03.01-2015.06.30, 2015.11.01-2016.04.09
        data, train_data, val_data, test_data, all_timestamps = \
            load_BJdata(fpath='../PCRN/data/TaxiBJ-filt/', split=split_point, T=params['ts_oneday'])

    # select data
    tshape = data.shape
    week_p = tshape[0] // week_len
    week_s = tshape[0] % week_len
    use_len = week_p * (week_len)
    # self
    self_data = data[-use_len:-week_len]
    self_data = np.vstack([self_data[:week_len], self_data])
    if week_s != 0:
        self_data = np.vstack([self_data[week_len-week_s:week_len], self_data])
    # forward 1 tslot
    pred_data = data[-use_len-1:-week_len-1]
    pred_data = np.vstack([pred_data[:week_len], pred_data])
    if week_s == 0:
        pred_data = np.vstack([pred_data[:1], pred_data])
    if week_s != 0:
        pred_data = np.vstack([pred_data[week_len-week_s:week_len], pred_data])
    # backward 1 tslot
    post_data = data[-use_len+1:-week_len+1]
    post_data = np.vstack([post_data[:week_len], post_data])
    if week_s != 0:
        post_data = np.vstack([post_data[week_len-week_s:week_len], post_data])
    # forward 1 week
    week_data = data[-use_len:-week_len*2]
    week_data = np.vstack([week_data[:week_len*2], week_data])
    if week_s != 0:
        week_data = np.vstack([week_data[week_len-week_s:week_len], week_data])
    # 8-nn
    kernel = np.ones([3,3]) * 1 / 8
    kernel[1,1] = 0
    knn_data = np.transpose(self_data, [0, 3, 1, 2])
    knn_data = np.array(
        list(map(lambda x:
                 list(map(lambda y: convolve(y, kernel, mode='nearest'), x)),
                 knn_data))
    )
    knn_data = np.transpose(knn_data, [0, 2, 3, 1])

    kde_result = np.sum(
        [self_data * 0.4,
         knn_data * 0.1,
         post_data * 0.1,
         pred_data * 0.1,
         week_data * 0.3
         ],
        axis=0
    )

    filt = self_data.transpose([3,0,1,2]).sum(axis=1)
    filt = np.where(filt<1e3, 0, 1).transpose([1,2,0])



    return np.multiply(kde_result, filt)


def kde_data(dataset='citybike'):
    if dataset in ['citybike', 'nyctaxi']:
        data, train_data, val_data, test_data = \
            load_npy_data(filename=['data/'+dataset+'/p_map.npy', 'data/'+dataset+'/d_map.npy'], split=split_point)
    else:
        # 2013.07.01-2013.10.29, 2014.03.01-2014.06.27, 2015.03.01-2015.06.30, 2015.11.01-2016.04.09
        data, train_data, val_data, test_data, all_timestamps = \
            load_BJdata(fpath='../PCRN/data/TaxiBJ-filt/', split=split_point, T=params['ts_oneday'])

    # select data
    tshape = data.shape
    week_p = tshape[0] // week_len
    week_s = tshape[0] % week_len
    use_len = week_p * (week_len)
    kde_data = data[-use_len:-week_len]
    kde_data = np.vstack([kde_data[:week_len], kde_data])
    if week_s != 0:
        kde_data = np.vstack([kde_data[week_len-week_s:week_len], kde_data])
    return kde_data



def hm_nycdata(dataset='citybike', hm=True):
    data, train_data, val_data, test_data = \
        load_npy_data(filename=['data/'+dataset+'/p_map.npy', 'data/'+dataset+'/d_map.npy'], split=split_point)

    # select data
    train_data = np.concatenate([train_data,val_data])
    tshape = train_data.shape
    use_len = tshape[0] // (week_len) * (week_len)
    train_data = train_data[-use_len:]

    # History Mean
    train_data = np.reshape(train_data, [train_data.shape[0]//(week_len), week_len, tshape[1], tshape[2], tshape[3]])
    hm = np.mean(train_data, axis=0)
    # hm = train_data[-1]

    week_p = data.shape[0] // week_len
    week_s = data.shape[0] % week_len
    hm_ext = np.array([hm]*week_p)
    tshape = hm_ext.shape
    hm_ext = np.reshape(hm_ext, [tshape[0]*tshape[1], tshape[2], tshape[3], tshape[4]])
    if week_s != 0:
        hm_ext = np.vstack([hm[-week_s:], hm_ext])

    return hm_ext


def hm_BJTaxi():
    # date range of data
    # 2013.07.01-2013.10.29, 2014.03.01-2014.06.27, 2015.03.01-2015.06.30, 2015.11.01-2016.04.09
    data, train_data, val_data, test_data, all_timestamps = \
        load_BJdata(fpath='../PCRN/data/TaxiBJ-filt/', split=split_point, T=48)

    train_data, test_data = data[:split_point['test']], data[split_point['test']:]

    # History Mean
    tshape = train_data.shape
    train_data = np.reshape(train_data, [tshape[0] // (week_len), week_len, tshape[1], tshape[2], tshape[3]])
    hm = np.mean(train_data, axis=0)

    # hm
    week_p = data.shape[0] // week_len
    week_s = data.shape[0] % week_len
    hm_ext = np.array([hm]*week_p)
    tshape = hm_ext.shape
    hm_ext = np.reshape(hm_ext, [tshape[0]*tshape[1], tshape[2], tshape[3], tshape[4]])
    if week_s != 0:
        hm_ext = np.vstack([hm[-week_s:], hm_ext])

    return hm_ext



def gen_timestamps_for_year(year):
    month1 = ['0'+str(e) for e in range(1,10)]
    month2 = [str(e) for e in range(10,13)]
    month = month1+month2
    day1 = ['0'+str(e) for e in range(1,10)]
    day2 = [str(e) for e in range(10,32)]
    day = day1+day2
    if year=='2012' or year=='2016':
        day_sum = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        day_sum = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    timestamps = []
    for m in range(len(month)):
        for d in range(day_sum[m]):
            t = [year+month[m]+day[d]]
            t_d = t*24
            timestamps.append(t_d[:])
    timestamps = np.hstack(np.array(timestamps))
    return timestamps

def gen_timestamps_for_year_ymdh(year):
    month1 = ['0'+str(e) for e in range(1,10)]
    month2 = [str(e) for e in range(10,13)]
    month = month1+month2
    day1 = ['0'+str(e) for e in range(1,10)]
    day2 = [str(e) for e in range(10,32)]
    day = day1+day2
    if year=='2012' or year=='2016':
        day_sum = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        day_sum = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hour1 = ['0'+str(e) for e in range(0,10)]
    hour2 = [str(e) for e in range(10,24)]
    hour = hour1+hour2
    timestamps = []
    for m in range(len(month)):
        for d in range(day_sum[m]):
            #t = [year+month[m]+day[d]]
            t_d = []
            for h in range(24):
                t_d.append(year+month[m]+day[d]+hour[h])
            timestamps.append(t_d[:])
    timestamps = np.hstack(np.array(timestamps))
    return timestamps

def gen_timestamps_for_year_ymdhm(year):
    month1 = ['0'+str(e) for e in range(1,10)]
    month2 = [str(e) for e in range(10,13)]
    month = month1+month2
    day1 = ['0'+str(e) for e in range(1,10)]
    day2 = [str(e) for e in range(10,32)]
    day = day1+day2
    if year=='2012' or year=='2016':
        day_sum = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        day_sum = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hour1 = ['0'+str(e) for e in range(0,10)]
    hour2 = [str(e) for e in range(10,24)]
    hour = hour1+hour2
    minute = ['00', '10', '20', '30', '40', '50']
    timestamps = []
    for m in range(len(month)):
        for d in range(day_sum[m]):
            #t = [year+month[m]+day[d]]
            t_d = []
            for h in range(24):
                a = [year+month[m]+day[d]+hour[h]+e for e in minute]
                #t_d = [t_d.append(year+month[m]+day[d]+hour[h]+e) for e in minute]
                t_d.append(a)
            t_d = np.hstack(np.array(t_d))
            timestamps.append(t_d[:])
    timestamps = np.hstack(np.array(timestamps))
    return timestamps

def gen_timestamps(years, gen_timestamps_for_year=gen_timestamps_for_year_ymdh):
    timestamps = []
    for y in years:
        timestamps.append(gen_timestamps_for_year(y))
    timestamps = np.hstack(np.array(timestamps))
    return timestamps

def shuffle_batch_data(data, batch_size=32, input_steps=10, output_steps=10):
    num = data.shape[0]
    # shuffle
    data = data[np.random.shuffle(np.arange(num)), :, :, :]

    x = []
    y = []
    i = 0
    while i<num-batch_size-input_steps-output_steps:
        batch_x = []
        batch_y = []
        for s in range(batch_size):
            batch_x.append(data[i+s:i+s+input_steps, :, :, :])
            batch_y.append(data[i+s+input_steps:i+s+input_steps+output_steps, :, :, :])
        x.append(batch_x)
        y.append(batch_y)
        i += batch_size
    return x, y

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print('Loaded %s..' %path)
        return file

def save_pickle(path,data):
    with open(path, 'rb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('Saved %s..' %path)
