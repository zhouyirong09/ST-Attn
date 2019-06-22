'''
Created on Apr, 2019

Multi-step Prediction Baseline: History Mean

@author: IronZhou
'''
from util.utils import *
from util.preprocessing import *
import tensorflow as tf
import pandas as pd
from util.params import *


def prepare_nycdata(dataset='citybike'):
    data, train_data, val_data, test_data = \
        load_npy_data(filename=['../data/'+dataset+'/p_map.npy', '../data/'+dataset+'/d_map.npy'], split=split_point)

    # select data
    train_data = np.concatenate([train_data,val_data])
    tshape = train_data.shape
    use_len = tshape[0] // (week_len) * (week_len)
    train_data = train_data[-use_len:]

    # History Mean
    train_data = np.reshape(train_data, [train_data.shape[0]//(week_len), week_len, tshape[1], tshape[2], tshape[3]])
    hm = np.mean(train_data, axis=0)
    # hm = train_data[-1]

    # build for test shape, extend as continue weeks
    # hm_ext = np.array([hm]*(4368//week_len))
    # tshape = hm_ext.shape
    # hm_ext = np.reshape(hm_ext, [tshape[0]*tshape[1], tshape[2], tshape[3], tshape[4]])
    return test_data, hm


def prepare_BJTaxi():
    # date range of data
    # 2013.07.01-2013.10.29, 2014.03.01-2014.06.27, 2015.03.01-2015.06.30, 2015.11.01-2016.04.09
    data, train_data, val_data, test_data, all_timestamps = \
        load_BJdata(fpath='../../PCRN/data/TaxiBJ-filt/', split=split_point, T=48)

    train_data, test_data = data[:split_point['test']], data[split_point['test']:]

    # History Mean
    tshape = train_data.shape
    train_data = np.reshape(train_data, [tshape[0] // (week_len), week_len, tshape[1], tshape[2], tshape[3]])
    hm = np.mean(train_data, axis=0)

    # build for test shape, extend as continue weeks
    # hm_ext = np.array([hm] * (test_data.shape[0] // week_len))
    # tshape = hm_ext.shape
    # hm_ext = np.reshape(hm_ext, [tshape[0] * tshape[1], tshape[2], tshape[3], tshape[4]])
    return test_data, hm


def cal_error(dataset, test_data, hm_ext):
    # test_data = pre_process.transform(test_data)
    hm_ext = np.array([hm_ext] * (test_data.shape[0] // week_len))
    tshape = hm_ext.shape
    hm_ext = np.reshape(hm_ext, [tshape[0] * tshape[1], tshape[2], tshape[3], tshape[4]])

    test_x, test_y = batch_data(data=test_data, batch_size=FLAGS.batch_size,
                                input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps)
    _, test_hm = batch_data(data=hm_ext, batch_size=FLAGS.batch_size,
                                input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps)

    test_y = np.vstack(test_y)
    test_hm = np.vstack(test_hm)
    np.save('../' + dataset + '-results/results/baseline_hm/prediction.npy', test_hm[-test_y.shape[0]:])
    print('RSME ERROR:', np.sqrt(np.mean(np.square(test_y - test_hm[-test_y.shape[0]:]))))
    print('MAE ERROR:', np.mean(np.abs(test_y - test_hm[-test_y.shape[0]:])))


if __name__ == '__main__':
    print('='*20,'\tCitiBike Data\t','='*20)
    test_data, hm_ext = prepare_nycdata(dataset='citybike')
    # hm_ext = kde_data2(dataset='citybike')
    cal_error('citybike', test_data, hm_ext)

    print('='*20,'\tNYCTaxi Data\t','='*20)
    test_data, hm_ext = prepare_nycdata(dataset='nyctaxi')
    # hm_ext = kde_data2(dataset='nyctaxi')
    cal_error('nyctaxi', test_data, hm_ext)

    # print('='*20,'\tBJTaxi Data\t','='*20)
    # test_data, hm_ext = prepare_BJTaxi()
    # hm_ext = kde_data2(dataset='BJTaxi')
    # cal_error('BJTaxi', test_data, hm_ext)