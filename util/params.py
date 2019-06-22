from imp_tf import *
import sys


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('input_steps', sys.argv[2],
                            """num of input_steps""")
tf.app.flags.DEFINE_integer('output_steps', sys.argv[3],
                            """num of output_steps""")
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """batch size for training""")
tf.app.flags.DEFINE_integer('n_epochs', 300,
                            """num of epochs""")
tf.app.flags.DEFINE_float('keep_prob', .9,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .00005,
                            """for dropout""")
tf.app.flags.DEFINE_string('update_rule', 'adam',
                            """update rule""")
tf.app.flags.DEFINE_integer('save_every', 1,
                            """steps to save""")
# model: ConvLSTM, AttConvLSTM, ResNet
tf.app.flags.DEFINE_string('model', sys.argv[1],
                            """which model to train and test""")
# ResNet
tf.app.flags.DEFINE_integer('closeness', 4,
                            """num of closeness""")
tf.app.flags.DEFINE_integer('period', 2,
                            """num of period""")
tf.app.flags.DEFINE_integer('trend', 1,
                            """num of trend""")
# AttConvLSTM
tf.app.flags.DEFINE_integer('cluster_num', 128,
                            """num of cluster in attention mechanism""")
tf.app.flags.DEFINE_integer('kmeans_run_num', 10,
                            """num of cluster in attention mechanism""")
tf.app.flags.DEFINE_integer('att_nodes', 1024,
                            """num of nodes in attention layer""")

params = {
    'ts_oneday': 48 if sys.argv[-1] in ['BJTaxi'] else 24, # or 48?
    'test_days': 21,
    'val_days': 21,
    'ext_l': 8,
    'ResNet': 8,
    'ConvLSTM':{
        'ec_conv':[16, 16],
        'ec_lstm':16,
        'dc_lstm':16,
        'dc_conv':16
    },
    'AttConvLSTM':{
        'attn_conv':[16, 16]
    },
    'ST-Attn':{
        'num_heads':8,
        'num_layers':2,
        's_attn_dim':16,
        't_attn_dim':64,
    },
    'input_steps': FLAGS.input_steps,
    'output_steps': FLAGS.output_steps
}

week_len = 7 * params['ts_oneday']

split_point = {
    'test': - params['test_days'] * params['ts_oneday'],
    'val': - (params['val_days']+params['val_days']) * params['ts_oneday'],
}