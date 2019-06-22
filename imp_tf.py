#!/usr/bin/python
# -*- coding:utf-8-*-

# from pyspark import SparkContext, SparkConf
# from pyspark.sql import SQLContext
#
#
# conf = SparkConf().setAppName("App")
# conf = (conf.setMaster("local[*]")
# 		.set('spark.executor.memory','14g')
# 		.set('spark.driver.memory','12g')
#         .set('spark.driver.maxresultsize', '2g')
#         .set('spark.driver.cores',8))
# sc = SparkContext(conf=conf)
# sqlContext = SQLContext(sc)



# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print('end')
