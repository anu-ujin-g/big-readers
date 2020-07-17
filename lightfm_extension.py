'''
Usage:
The program can be by typing the following command from the shell promt:

	park-submit lightfm_extension.py [1 or 5 or 25 or 100]

where 1 or 5 or 25 or 1000 are the percentage of dataset you would like to downsample 
'''



import sys
import pyspark
from pyspark.sql import SparkSession
from lightfm import LightFM 
from lightfm.evaluation import precision_at_k, auc_score
from sklearn import preprocessing

import pandas as pd 
import numpy as np 
from scipy.sparse import coo_matrix
import time




def load_data(spark, downsample):
	

	print('*** reading parquet ***')
	if downsample == 1:
		train = spark.read.parquet("train_set1.parquet")
		test = spark.read.parquet("test_set1.parquet")
	elif downsample == 5: 
		train = spark.read.parquet("train_set5.parquet")
		test = spark.read.parquet("test_set5.parquet")
	elif downsample == 25:
		train = spark.read.parquet("train_set25.parquet")
		test = spark.read.parquet("test_set25.parquet")
	else: 
		train = spark.read.parquet("train_set.parquet")
		test = spark.read.parquet("test_set.parquet") 

	# Drop is_read and is_reviewed columns
	cols_to_drop = ['is_read', 'is_reviewed']
	train = train.drop(*cols_to_drop)
	test = test.drop(*cols_to_drop) 
	
	print('*** converting to panda df ***')
	# convert spark df to pandas df
	train_pd = train.toPandas()
	test_pd = test.toPandas()

	return (train_pd, test_pd)

''' code taken from 
https://towardsdatascience.com/if-you-cant-measure-it-you-can-t-improve-it-5c059014faad'''
def convert_to_coo_matrix(train, test):
	
	test = test[(test['user_id'].isin(train['user_id'])) & (test['book_id'].isin(train['book_id']))]
	id_cols = ['user_id', 'book_id']
	trans_cat_train = dict()
	trans_cat_test = dict()

	print('Entering loop')
	for i in id_cols:
		cate_enc = preprocessing.LabelEncoder()
		trans_cat_train[i] = cate_enc.fit_transform(train[i].values)
		trans_cat_test[i] = cate_enc.transform(test[i].values)

	print('do ratings')
	cate_enc = preprocessing.LabelEncoder()
	ratings = dict()
	ratings['train'] = cate_enc.fit_transform(train.rating)
	ratings['test'] = cate_enc.transform(test.rating)

	print('getting n_users & n_items')
	n_users = len(np.unique(trans_cat_train['user_id']))
	n_items = len(np.unique(trans_cat_train['book_id']))

	print('converting to coo')
	train_coo = coo_matrix((ratings['train'], (trans_cat_train['user_id'], trans_cat_train['book_id'])), shape=(n_users, n_items))
	test_coo = coo_matrix((ratings['test'], (trans_cat_test['user_id'], trans_cat_test['book_id'])), shape=(n_users, n_items))

	return (train_coo, test_coo)


''' code taken from 
https://towardsdatascience.com/if-you-cant-measure-it-you-can-t-improve-it-5c059014faad'''
def lightfm_rec(train, test):
	
	print('!! enter lightfm !!')
	start = time.time()
	print('creating lightfm model and fitting')
	model = LightFM(k=500, item_alpha=0.75, user_alpha=0.75, loss='warp')
	model.fit(train, epochs=5, num_threads=4)
	print('Model fitting time =', time.time()-start)
	# print('calculating auc......')
	# auc_test = auc_score(model, test).mean()
	print('calculating pak......')
	pak_test = precision_at_k(model, test, k=500).mean()
	# print('auc =', auc_test)
	print('precision at k =', pak_test)
	print('total time =', time.time()-start)
	return (model, pak_test)




if __name__ == "__main__":
	memory = "5g"
	spark = SparkSession.builder \
                  .appName('lightfm') \
                  .master('yarn') \
                  .config('spark.executor.memory', memory) \
                  .config('spark.driver.memory', memory) \
                  .config('spark.executor.memoryOverhead', memory) \
                  .config("spark.sql.broadcastTimeout", "36000") \
                  .config("spark.storage.memoryFraction","0") \
                  .config("spark.memory.offHeap.enabled","true") \
                  .config("spark.memory.offHeap.size",memory) \
                  .getOrCreate()



	downsample = sys.argv[1]
	train, test = load_data(spark, downsample)
	train_coo, test_coo = convert_to_coo_matrix(train, test)
	model, pak = lightfm_rec(train_coo, test_coo)




