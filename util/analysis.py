import pandas as pd
import numpy as np
import os, glob
from imp_tf import *
from matplotlib import pyplot as plt
from util.params import params
from util.utils import *


# dataset = 'BJTaxi'
# dataset = 'citybike'
# dataset = 'nyctaxi'
preprocess_max = {
    'BJTaxi':1292.0,
    'citybike':526.0,
    'nyctaxi':9806.0
}
# models = ['ResNet', 'UNet', 'ConvLSTM', 'AttConvLSTM', 'ST-Attn', 'ConvLSTM_bi']
models = [
    # 'ResNet',
    # 'UNet',
    # 'UNet_Jump/UNet',
    # 'UNet_Jump/UNet_split',
    # 'ConvLSTM',
    # 'ConvLSTM_bi',
    'AttConvLSTM',
    'PCRN',
    'ST-Attn',
    # 'ST-Attn_y',
    # 'ST-Attn_hm'
    'baseline_hm',
    'baseline_kde'
]


def jump_last(input, results_path, dataset, split=False):
    a = np.array(input).reshape([-1, params['output_steps']])
    if split == True:
        trgt = np.vstack(np.load(results_path+'UNet_Jump/UNet_split/target.npy'))
        fp = glob.glob(results_path+'UNet_Jump/UNet_split/prediction*.npy')
        fp.sort()
        error = []
        for i in range(int(len(fp)/6)):
            result = []
            for f in fp[i*6:(i+1)*6]:
                result.append(np.vstack(np.load(f)))
            prediction = np.stack(result, axis=1)
            prediction = np.vstack(prediction[-int(trgt.shape[0]/6):])
            error.append(np.sqrt(np.mean(np.square(trgt-prediction))) * preprocess_max[dataset])
            # error.append(np.mean(np.abs(trgt-prediction)) * preprocess_max[dataset])
        # return list(np.mean(a, axis=1))
        return error
    else:
        return list(a[:,-1])

def whole_performance(results_path, dataset):
    results_dict = {}
    for m in models:
        results_dict[m] = {
            'train_loss':[],
            'val_loss':[],
            'test_loss':[]
        }
        fp = glob.glob(results_path+m+'/*.csv')
        fp.sort()
        for f in fp:
            data = pd.read_csv(f)
            results_dict[m]['train_loss'].append(data[data['loss_type']=='train_loss']['loss'].values[-1])
            results_dict[m]['val_loss'].append(data[data['loss_type']=='val_loss']['loss'].values[-1])
            results_dict[m]['test_loss'].append(data[data['loss_type']=='test_loss']['loss'].values[-1])
        if 'Jump' in m:
            results_dict[m]['train_loss'] = jump_last(results_dict[m]['train_loss'], results_path, dataset, split='split' in m)
            results_dict[m]['val_loss'] = jump_last(results_dict[m]['val_loss'], results_path, dataset, split='split' in m)
            results_dict[m]['test_loss'] = jump_last(results_dict[m]['test_loss'], results_path, dataset, split='split' in m)


    train_loss = [np.array(results_dict[k]['train_loss'], dtype=float) for k in results_dict]
    val_loss = [np.array(results_dict[k]['val_loss'], dtype=float) for k in results_dict]
    test_loss = [np.array(results_dict[k]['test_loss'], dtype=float) for k in results_dict]
    keys = [item for item in results_dict]
    # plt.figure(figsize=(15,6))
    # plt.boxplot(train_loss, labels=keys)
    # plt.grid()
    # plt.show()
    # plt.figure(figsize=(15,6))
    # plt.boxplot(val_loss, labels=keys)
    # plt.grid()
    # plt.show()
    plt.figure(figsize=(15,6))
    plt.boxplot(test_loss, labels=keys)
    plt.grid()
    plt.show()
    return [np.round(np.array(results_dict[k]['test_loss'], dtype=float).mean(),3) for k in results_dict]


def each_step(results_path, dataset):
    results = []
    for m in models:
        # m = 'ConvLSTM'
        print(m)
        trgt = np.vstack(np.load(results_path + 'ST-Attn/target.npy'))
        fp = glob.glob(results_path + m + '/prediction*.npy')
        fp.sort()
        if 'Jump' in m and 'split' not in m:
            fp = np.array(fp).reshape([-1, 6])[:, -1].tolist()
        pred = []

        for f in fp:
            if m in ['ResNet', 'UNet']:
                pred.append(np.load(f))
            elif m == 'PCRN':
                pred.append(np.transpose(np.load(f), [1,2,3,4,0]))
            elif 'Jump' in m and 'split' not in m:
                pred.append(np.vstack(np.load(f)))
            elif 'Jump' in m and 'split' in m:
                pred.append(np.vstack(np.load(f)))
            else:
                pred.append(np.vstack(np.load(f)))
        if 'Jump' in m and 'split' in m:
            predtmp = []
            for i in range(int(len(pred)/6)):
                predtmp.append(np.stack(pred[i*6:(i+1)*6], axis=1))
            pred = predtmp
        # pred = np.mean(pred, axis=0)
        PRECISION = []
        for item in pred:
            xshape = item.shape
            tslots = np.min([xshape[0], trgt.shape[0]])
            precision = []
            for i in range(6):
                if m == 'PCRN':
                    precision.append(np.sqrt(np.mean(np.square(trgt[-tslots:, i, :,:,:]* preprocess_max[dataset]-item[-tslots:, i, :,:,:]))))
                else:
                    precision.append(
                        np.sqrt(np.mean(np.square(trgt[-tslots:, i, :,:,:] - item[-tslots:, i, :,:,:]))) * preprocess_max[dataset])
            PRECISION.append(precision)
        precision = np.round(np.mean(PRECISION, axis=0),3)
        results.append(precision)
    results_df = pd.DataFrame(results, columns=['step'+str(i+1) for i in range(6)])
    results_df.to_csv('../result-collect/'+dataset+'.csv', index=False)


def abnormal_test(results_path, dataset):
    if dataset in ['BJTaxi', 'citybike', 'nyctaxi']:
        hm_ext = kde_data2(dataset)
    else:
        hm_ext = hm_nycdata(dataset='citybike')
    _, test_hm = batch_data(data=hm_ext, batch_size=FLAGS.batch_size,
                            input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps)
    test_hm = np.vstack(test_hm[-20:])

    results = []
    for m in models:
        # m = 'baseline_hm'
        print(m)
        trgt = np.vstack(np.load(results_path + 'ST-Attn/target.npy'))
        fp = glob.glob(results_path + m + '/predict*.npy')
        fp.sort()
        if 'Jump' in m and 'split' not in m:
            fp = np.array(fp).reshape([-1, 6])[:, -1].tolist()
        pred = []

        for f in fp:
            if m in ['ResNet', 'UNet']:
                pred.append(np.load(f))
            elif m == 'PCRN':
                pred.append(np.transpose(np.load(f), [1,2,3,4,0]))
            elif 'Jump' in m and 'split' not in m:
                pred.append(np.vstack(np.load(f)))
            elif 'Jump' in m and 'split' in m:
                pred.append(np.vstack(np.load(f)))
            elif 'baseline' in m:
                pred.append(np.load(f))
            else:
                pred.append(np.vstack(np.load(f)))
        if 'Jump' in m and 'split' in m:
            predtmp = []
            for i in range(int(len(pred)/6)):
                predtmp.append(np.stack(pred[i*6:(i+1)*6], axis=1))
            pred = predtmp
        # pred = np.mean(pred, axis=0)
        PRECISION = []
        for item in pred:

            xshape = item.shape
            tslots = np.min([xshape[0], trgt.shape[0]])
            abnormals = np.abs(trgt[-tslots:]-test_hm[-tslots:]) / (test_hm[-tslots:]+1)
            abnormals = np.where(abnormals < 0.5, 0, 1)
            precision = []
            for i in range(6):
                if m == 'PCRN':
                    precision.append(np.sqrt(1/abnormals[:,i].sum()*np.sum(abnormals[:,i]*np.square(trgt[-tslots:, i, :,:,:]* preprocess_max[dataset]-item[-tslots:, i, :,:,:]))))
                elif 'baseline' in m:  
                    precision.append(
                        np.sqrt(1/abnormals[:,i].sum()*np.sum(abnormals[:,i]*np.square(trgt[-tslots:, i, :,:,:]* preprocess_max[dataset] - item[-tslots:, i, :,:,:]))) )
                else:
                    precision.append(
                        np.sqrt(1/abnormals[:,i].sum()*np.sum(abnormals[:,i]*np.square(trgt[-tslots:, i, :,:,:]* preprocess_max[dataset] - item[-tslots:, i, :,:,:]))) )
            PRECISION.append(precision)
        precision = np.round(np.mean(PRECISION, axis=0),3)
        results.append(precision)
    results_df = pd.DataFrame(results, columns=['step'+str(i+1) for i in range(6)])
    results_df.to_csv('../result-collect/'+dataset+'_abnormal.csv', index=False)


def instance(results_path, dataset):
    results = []
    for m in models:
        # m = 'ConvLSTM'
        print(m)
        trgt = np.vstack(np.load(results_path + 'ST-Attn/target.npy'))
        fp = glob.glob(results_path + m + '/prediction*.npy')
        fp.sort()
        if 'Jump' in m and 'split' not in m:
            fp = np.array(fp).reshape([-1, 6])[:, -1].tolist()
        pred = []

        for f in fp:
            if m in ['ResNet', 'UNet']:
                pred.append(np.load(f))
            elif m == 'PCRN':
                pred.append(np.transpose(np.load(f), [1,2,3,4,0]))
            elif 'Jump' in m and 'split' not in m:
                pred.append(np.vstack(np.load(f)))
            elif 'Jump' in m and 'split' in m:
                pred.append(np.vstack(np.load(f)))
            else:
                pred.append(np.vstack(np.load(f)))
        if 'Jump' in m and 'split' in m:
            predtmp = []
            for i in range(int(len(pred)/6)):
                predtmp.append(np.stack(pred[i*6:(i+1)*6], axis=1))
            pred = predtmp
        # pred = np.mean(pred, axis=0)
        trgt_12 = np.concatenate([trgt[-week_len+24:-week_len+24+7,:,16,16,0][:-1,0],
                                 trgt[-week_len+24:-week_len+24 + 7, :, 16, 16, 0][-1, :]]) * preprocess_max[dataset]
        pred_12 = np.concatenate([trgt[-week_len+24:-week_len+24+7,:,16,16,0][:-1,0],
                                 pred[0][-week_len+24:-week_len+24 + 7, :, 16, 16, 0][-1, :]])
        if m != 'PCRN':
            pred_12 = pred_12 * preprocess_max[dataset]
        results.append(pred_12)
    results.append(trgt_12)
    a = np.array(results)
    plt.plot(a.transpose())
    plt.show()


if __name__ == '__main__':
    # results = []
    # for dataset in ['BJTaxi', 'nyctaxi', 'citybike']:
    #     results_path = '/cluster/zhouyirong09/peer-work/ST-Attn/result-collect/' + dataset + '/'
    #     results.append(whole_performance(results_path, dataset))
    #
    # result_df = pd.DataFrame(results, columns=models)
    # result_df.to_csv('../result-collect/average.csv', index=False)

    results = []
    for dataset in ['BJTaxi', 'nyctaxi', 'citybike']:#
        results_path = '/cluster/zhouyirong09/peer-work/ST-Attn/result-collect/' + dataset + '/'
        # instance(results_path, dataset)
        abnormal_test(results_path, dataset)
        # each_step(results_path, dataset)