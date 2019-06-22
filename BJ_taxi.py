import numpy as np
from imp_tf import *
import sys, os
from sklearn.cluster import KMeans
from solver_early import ModelSolver
from util.preprocessing import *
from util.utils import *
from util.params import *
import pandas as pd


dataset = sys.argv[-1] # 'BJTaxi'
model_used = FLAGS.model # 'AttConvLSTM' # ResNet, UNet, ConvLSTM, AttConvLSTM, ST-Attn, ConvLSTM_bi
print('model training:', model_used)
print('on dataset:', dataset)


start_t = time.time()


def main():
    # preprocessing class
    pre_process = MinMaxNormalization01()
    print('load train, validate, test data...')
    # date range of data
    if dataset == 'BJTaxi':
        # 2013.07.01-2013.10.29, 2014.03.01-2014.06.27, 2015.03.01-2015.06.30, 2015.11.01-2016.04.09
        data, train_data, val_data, test_data, all_timestamps = \
            load_BJdata(fpath='../PCRN/data/TaxiBJ-filt/',split=split_point, T=params['ts_oneday'])
        hm_ext = hm_BJTaxi()
        kde_ext = kde_data2(dataset=dataset)
    elif dataset == 'citybike':
        data, train_data, val_data, test_data = \
            load_npy_data(filename=['data/citybike/p_map.npy', 'data/citybike/d_map.npy'], split=split_point)
        all_timestamps = gen_timestamps(['2013', '2014', '2015', '2016'])
        # 2013-07-01, 2016-06-30
        all_timestamps = all_timestamps[4344:-4416]
        hm_ext = hm_nycdata(dataset=dataset)
        kde_ext = kde_data2(dataset=dataset)
    elif dataset == 'nyctaxi':
        data, train_data, val_data, test_data = \
            load_npy_data(filename=['data/nyctaxi/p_map.npy', 'data/nyctaxi/d_map.npy'], split=split_point)
        all_timestamps = gen_timestamps(['2014', '2015', '2016'])
        # 2014-01-01, 2016-06-30
        all_timestamps = all_timestamps[:-4416]
        hm_ext = hm_nycdata(dataset=dataset)
        kde_ext = kde_data2(dataset=dataset)
    else:
        print('Please chosse dataset in BJTaxi, citybike, nyctaxi!')
        return

    # data: [num, row, col, channel]
    print('preprocess train data...')
    pre_process.fit(train_data)
    # use timestamps as external features
    all_timestamps = external_feature2(all_timestamps)
    ext_train, ext_val, ext_test = prepare_all(
        all_timestamps[:split_point['val']],
        all_timestamps[split_point['val']:split_point['test']],
        all_timestamps[split_point['test']:],
        pre_process=None
    )
    # train, val, test dataset
    train, val, test = prepare_all(train_data, val_data, test_data, pre_process)

    # data shape
    nb_flow = train_data.shape[-1]
    row = train_data.shape[1]
    col = train_data.shape[2]

    input_dim = [row, col, nb_flow]

    if FLAGS.model == 'ST-Attn_xy':
        # input of encoder: (x+hm) / 2
        # input of decoder: hm
        pre_process1 = MinMaxNormalization01()
        pre_process1.fit(hm_ext)
        hm_train, hm_val, hm_test = prepare_all(
            hm_ext[:split_point['val']],
            hm_ext[split_point['val']:split_point['test']],
            hm_ext[split_point['test']:],
            pre_process=pre_process1
        )
        from model.ST_Attn_xy import ST_Attn
        model = ST_Attn(
            input_dim=input_dim,
            batch_size=FLAGS.batch_size,
            params=params['ST-Attn'],
            input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps,
            ext_data={'train':[ext_train['x'], ext_train['y']],
                      'val':[ext_val['x'],  ext_val['y']],
                      'test':[ext_test['x'], ext_test['y']]},
            hm_data={'train':[hm_train['x'], hm_train['y']],
                      'val':[hm_val['x'],  hm_val['y']],
                      'test':[hm_test['x'], hm_test['y']]}
        )
    elif FLAGS.model == 'ST-Attn_PC':
        # input of decoder: x, k_means results
        # input of decoder: y
        pre_process1 = MinMaxNormalization01()
        pre_process1.fit(hm_ext)
        hm_train, hm_val, hm_test = prepare_all(
            hm_ext[:split_point['val']],
            hm_ext[split_point['val']:split_point['test']],
            hm_ext[split_point['test']:],
            pre_process=pre_process
        )

        # K-Means
        cluster_fp = 'cluster-results/'+dataset+'/cluster.npy'
        if os.path.isfile(cluster_fp):
            cluster_centroid = np.load(cluster_fp)
        else:
            pc = np.vstack(train['y']).reshape([-1, FLAGS.output_steps, row, col, nb_flow])
            vector_data = np.reshape(pc, (pc.shape[0], -1))
            kmeans = KMeans(n_clusters=FLAGS.cluster_num, init='random', n_init=FLAGS.kmeans_run_num, tol=0.00000001).fit(
                vector_data)
            cluster_centroid = kmeans.cluster_centers_
            cluster_centroid = np.reshape(cluster_centroid,
                                          (-1, FLAGS.output_steps, row, col, nb_flow))
            np.save(cluster_fp, cluster_centroid)

        from model.ST_Attn_PC import ST_Attn
        model = ST_Attn(
            input_dim=input_dim,
            batch_size=FLAGS.batch_size,
            params=params['ST-Attn'],
            input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps,
            pc_data=cluster_centroid,
            ext_data={'train':[ext_train['x'], ext_train['y']],
                      'val':[ext_val['x'],  ext_val['y']],
                      'test':[ext_test['x'], ext_test['y']]},
            hm_data={'train':[hm_train['x'], hm_train['y']],
                      'val':[hm_val['x'],  hm_val['y']],
                      'test':[hm_test['x'], hm_test['y']]}
        )
    elif FLAGS.model == 'ST-Attn_PCY':
        # input of decoder: x, k_means results
        # input of decoder: PC-Attn
        pre_process1 = MinMaxNormalization01()
        pre_process1.fit(hm_ext)
        hm_train, hm_val, hm_test = prepare_all(
            hm_ext[:split_point['val']],
            hm_ext[split_point['val']:split_point['test']],
            hm_ext[split_point['test']:],
            pre_process=pre_process1
        )

        # K-Means
        cluster_fp = 'cluster-results/'+dataset+'/cluster.npy'
        if os.path.isfile(cluster_fp):
            cluster_centroid = np.load(cluster_fp)
        else:
            pc = np.array(train['y']).reshape([-1, FLAGS.output_steps, row, col, nb_flow])
            vector_data = np.reshape(pc, (pc.shape[0], -1))
            kmeans = KMeans(n_clusters=FLAGS.cluster_num, init='random', n_init=FLAGS.kmeans_run_num, tol=0.00000001).fit(
                vector_data)
            cluster_centroid = kmeans.cluster_centers_
            cluster_centroid = np.reshape(cluster_centroid,
                                          (-1, FLAGS.output_steps, row, col, nb_flow))
            np.save(cluster_fp, cluster_centroid)

        from model.ST_Attn_PCY import ST_Attn
        model = ST_Attn(
            input_dim=input_dim,
            batch_size=FLAGS.batch_size,
            params=params['ST-Attn'],
            input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps,
            pc_data=cluster_centroid,
            ext_data={'train':[ext_train['x'], ext_train['y']],
                      'val':[ext_val['x'],  ext_val['y']],
                      'test':[ext_test['x'], ext_test['y']]},
            hm_data={'train':[hm_train['x'], hm_train['y']],
                      'val':[hm_val['x'],  hm_val['y']],
                      'test':[hm_test['x'], hm_test['y']]}
        )
    elif '-Attn_y' in FLAGS.model:
        # input of decoder: x
        # input of decoder: hm
        pre_process1 = MinMaxNormalization01()
        pre_process1.fit(hm_ext)
        hm_train, hm_val, hm_test = prepare_all(
            hm_ext[:split_point['val']],
            hm_ext[split_point['val']:split_point['test']],
            hm_ext[split_point['test']:],
            pre_process=pre_process1
        )
        pre_process2 = MinMaxNormalization01()
        pre_process2.fit(kde_ext)
        kde_train, kde_val, kde_test = prepare_all(
            kde_ext[:split_point['val']],
            kde_ext[split_point['val']:split_point['test']],
            kde_ext[split_point['test']:],
            pre_process=pre_process2
        )
        if 'S-Attn_y_poe' in FLAGS.model:
            from model.S_Attn_y_poe import ST_Attn
        if 'T-Attn_y_poe' in FLAGS.model:
            from model.T_Attn_y_poe import ST_Attn
        elif 'poea' in FLAGS.model:
            from model.ST_Attn_y_poea import ST_Attn
        elif 'poe' in FLAGS.model:
            from model.ST_Attn_y_poe import ST_Attn
        elif 'hm' in FLAGS.model:
            from model.ST_Attn_y_hm import ST_Attn
        else:
            from model.ST_Attn_y import ST_Attn
        model = ST_Attn(
            input_dim=input_dim,
            batch_size=FLAGS.batch_size,
            params=params['ST-Attn'],
            input_steps=FLAGS.input_steps, output_steps=FLAGS.output_steps,
            kde_data={'train':[kde_train['x'], kde_train['y']],
                      'val':[kde_val['x'],  kde_val['y']],
                      'test':[kde_test['x'], kde_test['y']]},
            hm_data={'train':[hm_train['x'], hm_train['y']],
                      'val':[hm_val['x'],  hm_val['y']],
                      'test':[hm_test['x'], hm_test['y']]}
        )
    else:
        return
    solver = ModelSolver(model, train, val, preprocessing=pre_process,
                         n_epochs=FLAGS.n_epochs,
                         batch_size=FLAGS.batch_size,
                         update_rule=FLAGS.update_rule,
                         learning_rate=FLAGS.lr, save_every=FLAGS.save_every,
                         pretrained_model=None, model_path=dataset+'-results/model_save/' + FLAGS.model + '/',
                         test_model=dataset+'-results/model_save/' + FLAGS.model + '/model-' + str(FLAGS.n_epochs),
                         log_path=dataset+'-results/log/' + FLAGS.model + '/')
    print('begin training...')
    test_prediction, df_result = solver.train(test)
    test_target = np.asarray(test['y'])


    if not os.path.exists(dataset+'-results/results/'+FLAGS.model+'/'):
        os.makedirs(dataset+'-results/results/'+FLAGS.model+'/')
    w_time = pd.datetime.now().strftime("%m%d%H%M")
    np.save(dataset+'-results/results/'+FLAGS.model+'/target.npy', test_target)
    np.save(dataset+'-results/results/'+FLAGS.model+'/prediction'+w_time+'.npy', test_prediction)
    params.update({'model_used': FLAGS.model})
    df_result.append([params, -1, time.time()-start_t])
    df_result = pd.DataFrame(df_result, columns=['loss_type','epoch','loss'])
    df_result.to_csv(dataset+'-results/results/'+FLAGS.model+'/'+w_time+'.csv', index=False)


if __name__ == "__main__":
    main()

