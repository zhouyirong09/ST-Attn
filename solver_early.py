#from __future__ import division
import numpy as np
import pandas as pd
import time
import os
from sklearn.model_selection import train_test_split
from imp_tf import *
import sys
sys.path.append('./util/')
from util import *


os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[4]
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9


class ModelSolver(object):
	def __init__(self, model, data, val_data, preprocessing, **kwargs):
		self.model = model
		self.data = data
		self.val_data = val_data
		self.preprocessing = preprocessing
		self.cross_val = kwargs.pop('cross_val', False)
		self.cpt_ext = kwargs.pop('cpt_ext', False)
		self.n_epochs = kwargs.pop('n_epochs', 10)
		self.batch_size = kwargs.pop('batch_size', 32)
		self.learning_rate = kwargs.pop('learning_rate', 0.000001)
		self.update_rule = kwargs.pop('update_rule', 'adam')
		self.model_path = kwargs.pop('model_path', './model/')
		self.save_every = kwargs.pop('save_every', 1)
		self.log_path = kwargs.pop('log_path', './log/')
		self.pretrained_model = kwargs.pop('pretrained_model', None)
		self.test_model = kwargs.pop('test_model', './model/lstm/model-1')

		# early_stopping control
		self.min_val = [0, preprocessing.get_max()]
		self.early_stop = 50 # 10 steps if val_loss does not improve

		if self.update_rule == 'adam':
			self.optimizer = tf.train.AdamOptimizer
		elif self.update_rule == 'momentum':
			self.optimizer = tf.train.MomentumOptimizer
		elif self.update_rule == 'rmsprop':
			self.optimizer = tf.train.RMSPropOptimizer

		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)
		if not os.path.exists(self.log_path):
			os.makedirs(self.log_path)

	def train(self, test_data, test_1_to_n_data=[], output_steps=10, ST_Attn=False):
		raw_x = x = self.data['x']
		raw_y = y = self.data['y']
		x_val = self.val_data['x']
		y_val = self.val_data['y']

		# build graphs
		y_, loss = self.model.build_model()

		# train op
		with tf.name_scope('optimizer'):
			optimizer = self.optimizer() if self.update_rule == 'adam' else self.optimizer(learning_rate=self.learning_rate)
			grads = tf.gradients(loss, tf.trainable_variables())
			grads_and_vars = list(zip(grads, tf.trainable_variables()))
			train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

		# summary op
		tf.get_variable_scope().reuse_variables()
		tf.summary.scalar('batch_loss', loss)
		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name, var)
		for grad, var in grads_and_vars:
			tf.summary.histogram(var.op.name+'/gradient', grad)
		summary_op = tf.summary.merge_all()
		# record training process to csv
		df_result = []

		with tf.Session(config=config) as sess:
			tf.global_variables_initializer().run()
			# summary_writer = tf.summary.FileWriter(self.log_path, graph=sess.graph)
			saver = tf.train.Saver(tf.global_variables())
			if self.pretrained_model is not None:
				print("Start training with pretrained model...")
				# saver.restore(sess, self.pretrained_model)

			#curr_loss = 0
			start_t = time.time()
			for e in range(self.n_epochs):
				self.model.bn_training = True
				print('epochs: ', e)
				curr_loss = 0

				if self.cross_val: # cross validation
					x, x_val, y, y_val = train_test_split(raw_x, raw_y, test_size=0.1, random_state=50)

				for i in range(len(x)):
					feed_dict = {
						self.model.x: np.array(x[i]),
						self.model.y: np.array(y[i]),
						self.model.xh: self.model.hm_data['train'][0][i],
						self.model.yh: self.model.hm_data['train'][1][i],
						self.model.dropout_rate: [0.5],
						self.model.xz: self.model.kde_data['train'][0][i],
						self.model.yz: self.model.kde_data['train'][1][i]
					}
					_, l = sess.run([train_op, loss], feed_dict)
					curr_loss += l

					# write summary for tensorboard visualization
					if i % 100 == 0:
						# summary = sess.run(summary_op, feed_dict)
						# summary_writer.add_summary(summary, e*len(x) + i)
						print("at epoch "+str(e)+', '+str(i))

				# compute counts of all regions
				t_count = 0
				for c in range(len(y)):
					#print(np.array(y[c]).shape)
					t_count += np.prod(np.array(y[c]).shape)
				t_rmse = np.sqrt(curr_loss/t_count)
				#t_rmse = np.sqrt(curr_loss/(np.prod(np.array(y).shape)))
				print("at epoch " + str(e) + ", train loss is " + str(curr_loss) + ' , ' + str(t_rmse) + ' , ' + str(self.preprocessing.real_loss(t_rmse)))
				df_result.append(['train_loss', e, self.preprocessing.real_loss(t_rmse)])
				# validate
				val_loss = 0

				self.model.bn_training = False
				for i in range(len(y_val)):
					feed_dict = {
						self.model.x: np.array(x_val[i]),
						self.model.y: np.array(y_val[i]),
						self.model.xh: self.model.hm_data['val'][0][i],
						self.model.yh: self.model.hm_data['val'][1][i],
						self.model.dropout_rate: [1.0],
						self.model.xz: self.model.kde_data['val'][0][i],
						self.model.yz: self.model.kde_data['val'][1][i]
					}
					_, l = sess.run([y_, loss], feed_dict=feed_dict)
					val_loss += l

				# y_val : [batches, batch_size, seq_length, row, col, channel]
				print(np.array(y_val).shape)
				v_count = 0
				for v in range(len(y_val)):
					v_count += np.prod(np.array(y_val[v]).shape)
				rmse = np.sqrt(val_loss/v_count)

				print("at epoch " + str(e) + ", validate loss is " + str(val_loss) + ' , ' + str(rmse) + ' , ' + str(self.preprocessing.real_loss(rmse)))
				df_result.append(['val_loss', e, self.preprocessing.real_loss(rmse)])
				print("elapsed time: ", time.time() - start_t)

				if self.preprocessing.real_loss(rmse) < self.min_val[1]:
					# update
					self.min_val = [e, self.preprocessing.real_loss(rmse)]
					save_name = self.model_path+'model'
					saver.save(sess, save_name)
					print("model-%s saved." % (e+1))

				# elif e - self.early_stop >= self.min_val[0]:
				# 	break
			# ============================ for test data ===============================
			print('test for test data...')
			x_test = test_data['x']
			y_test = test_data['y']
			t_loss = 0
			y_pre_test = []
			saver.restore(sess, self.model_path+'model')

			self.model.bn_training = False
			for i in range(len(y_test)):
				feed_dict = {
					self.model.x: np.array(x_test[i]),
					self.model.y: np.array(y_test[i]),
					self.model.xh: self.model.hm_data['test'][0][i],
					self.model.yh: self.model.hm_data['test'][1][i],
					self.model.dropout_rate: [1.0],
					self.model.xz: self.model.kde_data['test'][0][i],
					self.model.yz: self.model.kde_data['test'][1][i]
				}
				y_pre_i, l = sess.run([y_, loss], feed_dict=feed_dict)
				t_loss += l
				y_pre_test.append(y_pre_i)

			# y_val : [batches, batch_size, seq_length, row, col, channel]
			print(np.array(y_test).shape)
			t_count = 0
			for t in range(len(y_test)):
				t_count += np.prod(np.array(y_test[t]).shape)
			print('t_count = '+str(t_count))
			rmse = np.sqrt(t_loss/t_count)

			print("at epoch " + str(self.min_val[0]) + ", test loss is " + str(t_loss) + ' , ' + str(rmse) + ' , ' + str(self.preprocessing.real_loss(rmse)))
			df_result.append(['test_loss', self.min_val[0], self.preprocessing.real_loss(rmse)])

			y_pre_test = np.asarray(y_pre_test)

			return y_pre_test, df_result

