from __future__ import division

from imp_tf import *
from functools import reduce
from operator import mul


class ST_Attn(object):
    def __init__(self, input_dim=[64, 64, 2], att_inputs=[], att_nodes=1024, batch_size=32,
                 input_steps=10, output_steps=10, ext_dim=8,
                 ext_data={}, hm_data={}, params={}):
        # self.input_dim = input_dim
        self.input_row = input_dim[0]
        self.input_col = input_dim[1]
        self.input_channel = input_dim[2]
        self.ext_data = ext_data
        self.hm_data = hm_data
        self.NUM_HEADS = params['num_heads']
        self.NUM_LAYERS = params['num_layers']
        self.s_attn_dim = params['s_attn_dim']
        self.t_attn_dim = params['t_attn_dim']

        self.batch_size = batch_size
        self.seq_length = input_steps + output_steps
        self.input_steps = input_steps
        self.output_steps = output_steps

        self.weight_initializer = tf.keras.initializers.glorot_normal(seed=None)
        # print('initializer: tf.contrib.layers.xavier_initializer()')
        self.const_initializer = tf.constant_initializer()

        self.x = tf.placeholder(tf.float32,
                                [None, self.input_steps, self.input_row, self.input_col, self.input_channel],
                                name='x')
        self.y = tf.placeholder(tf.float32,
                                [None, self.output_steps, self.input_row, self.input_col, self.input_channel],
                                name='y')
        self.xz = tf.placeholder(tf.float32,
                                [None, self.input_steps, self.input_row, self.input_col, self.input_channel],
                                name='xz')
        self.yz = tf.placeholder(tf.float32,
                                [None, self.output_steps, self.input_row, self.input_col, self.input_channel],
                                 name='z')
        self.yz_data = []
        self.dropout_rate = tf.placeholder(tf.float32, 1, name='dropout_rate')

        # for external input
        self.x_ext = tf.placeholder(tf.float32, [None, self.input_steps, ext_dim], name='x_ext')#self.input_conf[-1][0]])
        self.y_ext = tf.placeholder(tf.float32, [None, self.output_steps, ext_dim], name='y_ext')#self.input_conf[-1][0]])

    def _exchange_ST(self, x_s, x_t):
        with tf.variable_scope('exchange-ST'):
            # con-transform -- S
            xs_shape = x_s.get_shape()
            out = 'SA' in tf.get_variable_scope().name or 'EA' in tf.get_variable_scope().name
            steps = self.output_steps if out else self.input_steps
            x_st = tf.layers.conv2d(x_s, filters=steps, kernel_size=[1,1], name='Reshape-S')
            x_st = tf.transpose(x_st, [0,3,1,2])
            x_st = tf.reshape(x_st, shape=[-1, steps, xs_shape[1]*xs_shape[2]])
            # con-transform -- T
            xt_shape = x_t.get_shape()
            x_ts = tf.layers.conv1d(x_t, filters=self.input_row*self.input_col, kernel_size=1, name='Reshape-T')
            x_ts = tf.reshape(x_ts, shape=[-1, xt_shape[1], self.input_row, self.input_col])
            x_ts = tf.transpose(x_ts, [0,2,3,1])
            # concat
            x_s = tf.concat([x_s, x_ts], axis=-1)
            x_t = tf.concat([x_t, x_st], axis=-1)
            return x_s, x_t

    def _encoder(self, x_s, x_t):
        with tf.variable_scope('encoder'):
            # 1. add position-encoding on x_ext

            # change for later shape
            x_s = tf.layers.conv2d(x_s, filters=self.s_attn_dim, kernel_size=[1,1], name='Reshape-S')
            x_t = tf.layers.conv1d(x_t, filters=self.t_attn_dim, kernel_size=1, name='Reshape-T')

            # 2. encoder layer stack
            state = x_s, x_t
            for idx in range(self.NUM_LAYERS):
                state = self._encoderlayer(state, idx)
        return state

    def _encoderlayer(self, x, idx):
        x_s, x_t = x
        with tf.variable_scope('encoderlayer', reuse=(idx!=0)):
            # attention
            state_s = self._spatialAttn(x_s, x_s, x_s)
            state_t = self._temporalAttn(x_t, x_t, x_t)

            # exchange S-T
            state_s, state_t = self._exchange_ST(state_s, state_t)

            # feedforward
            state_s, state_t = self._positionwisefeedforward(state_s, state_t)

        return state_s, state_t

    def _decoder(self, state, y_ext):
        with tf.variable_scope('decoder'):
            # 1. add position-encoding on y_ext

            # 2. decoder layer stack
            x_shape = self.y.get_shape()

            y_s = tf.reshape(tf.transpose(self.yz, [0,2,3,1,4]),
                             shape=[-1, x_shape[2], x_shape[3], x_shape[1] * x_shape[-1]])
            y_t = tf.reshape(self.yz, [-1, x_shape[1], x_shape[2] * x_shape[3] * x_shape[4]])
            # change for later shape
            y_s = tf.layers.conv2d(y_s, filters=self.s_attn_dim, kernel_size=[1,1], name='Additional-S')
            y_t = tf.layers.conv1d(y_t, filters=self.t_attn_dim, kernel_size=1, name='Additional-T')
            y_ = y_s, y_t
            for idx in range(3):
                y_ = self._decoderlayer(y_, state, idx)

            # 3. Linear position-wise-feedforward
        print('decoder')
        return y_

    def _decoderlayer(self, y, state, idx):
        y_s, y_t = y
        state_s, state_t = state
        with tf.variable_scope('decoderlayer', reuse=(idx!=0)):
            with tf.variable_scope('SA'):
                # self-attention
                y_s = self._spatialAttn(y_s, y_s, y_s)
                y_t = self._temporalAttn(y_t, y_t, y_t)
            with tf.variable_scope('EA'):
                # encoder-attention
                y_s = self._spatialAttn(y_s, state_s, state_s)
                y_t = self._temporalAttn(y_t, state_t, state_t)
            # feedforward
            y_s, y_t = self._positionwisefeedforward(y_s, y_t)
            print('decoderlayer')
        return y_s, y_t

    def _multiheadattn(self, x):
        # whatever NUM_STACKS is, the conv-params only linear to NUM_HEADS
        with tf.variable_scope('multihead-attention', reuse=True):
            head_tensors = []
            for i in range(self.NUM_HEADS):
                # spatial
                head_tensors.append(x)
                # temporal
                head_tensors.append(x)

            head_cat = tf.stack(head_tensors)
            mha = self._conv(head_cat, 8, 'mha')
        return mha

    def _spatialAttn(self, x_s_q, x_s_k, x_s_v):
        with tf.variable_scope('S-Attn'):
            # conv-transform
            f = tf.nn.relu(tf.layers.batch_normalization(
                tf.layers.conv2d(x_s_q, self.s_attn_dim, kernel_size=[3, 3], padding='same', name='f',
                                 kernel_initializer=self.weight_initializer)))
            g = tf.nn.relu(tf.layers.batch_normalization(
                tf.layers.conv2d(x_s_k, self.s_attn_dim, kernel_size=[3, 3], padding='same', name='g',
                                 kernel_initializer=self.weight_initializer)))
            h = tf.nn.relu(tf.layers.batch_normalization(
                tf.layers.conv2d(x_s_v, self.s_attn_dim, kernel_size=[3, 3], padding='same', name='h',
                                 kernel_initializer=self.weight_initializer)))

            # f, g, h
            s = tf.matmul(self._hw_flatten(f), self._hw_flatten(g), transpose_b=True)

            # softmax(Q, K)*V
            beta = tf.nn.softmax(s, axis=-1)
            o = tf.matmul(beta, self._hw_flatten(h))
            # gama = tf.get_variable('gama', [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=[-1, h.shape[1], h.shape[2], h.shape[3]])
            # x = gama * o + x
            o = tf.layers.dropout(o, rate=self.dropout_rate)
            # x = self._layernorm(x_s_v + gama * o, dim=1)
            # x = self._layernorm(o, dim=1)
        return o

    def _temporalAttn(self, x_t_q, x_t_k, x_t_v):
        x_shape = x_t_q.get_shape()
        with tf.variable_scope('T-Attn'):
            # conv-transform
            f = tf.nn.relu(tf.layers.batch_normalization(
                tf.layers.conv1d(x_t_q, filters=self.t_attn_dim, kernel_size=3, padding='same', name='f',
                                 kernel_initializer=self.weight_initializer)))
            g = tf.nn.relu(tf.layers.batch_normalization(
                tf.layers.conv1d(x_t_k, filters=self.t_attn_dim, kernel_size=3, padding='same', name='g',
                                 kernel_initializer=self.weight_initializer)))
            h = tf.nn.relu(tf.layers.batch_normalization(
                tf.layers.conv1d(x_t_v, filters=self.t_attn_dim, kernel_size=3, padding='same', name='h',
                                 kernel_initializer=self.weight_initializer)))

            # f, g, h
            s = tf.matmul(f, g, transpose_b=True)

            # softmax(Q, K)*V
            beta = tf.nn.softmax(s, axis=-1)
            o = tf.matmul(beta, h)
            # gama = tf.get_variable('gama', [1], initializer=tf.constant_initializer(0.0))

            # o = tf.reshape(o, shape=[-1, x.shape[1], x.shape[2]])
            # x = gama * o + x_t_v
            o = tf.layers.dropout(o, rate=self.dropout_rate)
            # x = self._layernorm(x_t_v + gama * o, dim=1)
            # x = self._layernorm(o, dim=1)
        return o

    def _positionwisefeedforward(self, x_s, x_t):
        with tf.variable_scope('S-FF'):
            x_s = tf.layers.conv2d(x_s, self.s_attn_dim, [1,1], strides=[1,1],
                                   kernel_initializer=self.weight_initializer)
            x_s = tf.layers.batch_normalization(
                x_s + tf.layers.dropout(x_s, rate=self.dropout_rate),
                axis = -1)
        with tf.variable_scope('T-FF'):
            x_t = tf.layers.conv1d(x_t, self.t_attn_dim, kernel_size=1,
                                   kernel_initializer=self.weight_initializer)
            x_t = tf.layers.batch_normalization(x_t + tf.layers.dropout(x_t, rate=self.dropout_rate),
                                                axis = -1)
        return x_s, x_t

    def _conv(self, inputs, output_features, idx, filter=[3,3], strides=[1,1,1,1], padding='SAME'):
        # param: filter, strides, output_features
        with tf.variable_scope('conv_{0}'.format(idx)) as scope:
            in_channels = inputs.get_shape().as_list()[3]
            w = tf.get_variable('w', [filter[0], filter[1], in_channels, output_features],
                                initializer=self.weight_initializer)
            b = tf.get_variable('b', [output_features], initializer=self.const_initializer)
            y = tf.nn.conv2d(inputs, w, strides=strides, padding=padding)
            y_b = tf.nn.bias_add(y, b, name='wx_plus_b')
            y_relu = tf.nn.relu(y_b, name='out_conv_{0}'.format(idx))
            return y_relu

    def _hw_flatten(self, x):
        return tf.reshape(x, shape=[-1, x.shape[1]*x.shape[2], x.shape[-1]])

    def build_model(self):
        x = self.x
        y = self.y
        x_ext = self.x_ext
        y_ext = self.y_ext
        with tf.variable_scope('ST-Attn'):
            # TODO: position-encoding with x_ext
            x_shape = x.get_shape()
            y_shape = y.get_shape()
            x_s = tf.reshape(tf.transpose(x, [0, 2, 3, 1, 4]),
                           shape=[-1, x_shape[2], x_shape[3], x_shape[1] * x_shape[-1]])
            x_t = tf.reshape(x, [-1, x_shape[1], x_shape[2]*x_shape[3]*x_shape[4]])

            # use external features
            # y_ext = tf.layers.dropout(
            #     tf.layers.conv1d(y_ext, filters=y_shape[2]*y_shape[3]*y_shape[4], kernel_size=1),
            #     rate=self.dropout_rate)
            # y_ext = tf.reshape(y_ext, shape=[-1, self.output_steps, y_shape[2], y_shape[3], y_shape[4]])
            # encoder-decoder
            state = self._encoder(x_s, x_t)
            y_s, y_t = self._decoder(state, y_ext)

            # fusion space-time
            y_t = tf.layers.dropout(
                tf.layers.conv1d(y_t, filters=y_shape[2]*y_shape[3], kernel_size=1),
                rate=self.dropout_rate)
            y_t = tf.reshape(y_t, shape=[-1, y_shape[1], y_shape[2], y_shape[3]])
            y_t = tf.transpose(y_t, [0,2,3,1])
            y_ = tf.concat([y_s, y_t], axis=-1)
            y_ = tf.layers.dropout(
                tf.layers.conv2d(y_, filters=self.output_steps*y_shape[-1], kernel_size=1),
                rate=self.dropout_rate)
            y_ = tf.reshape(y_, shape=[-1, y_shape[2], y_shape[3], self.output_steps, y_shape[-1]])
            y_ = tf.transpose(y_, [0,3,1,2,4])

            # output
            # y_ = tf.nn.relu(tf.add(y_, y_ext), name='y_out')
            loss = 2 * tf.nn.l2_loss(y - y_[:, :, :, :, :])

        print('all params count up to:', self.get_num_params())
        return y_, loss

    def get_num_params(self):
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params