import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, desc
from pyspark.mllib.recommendation import Rating, MatrixFactorizationModel
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics, RegressionMetrics
import itertools


def train_model(spark):
	print("Loading parquet data")
	train = spark.read.parquet("train_set.parquet")
	val = spark.read.parquet("val_set.parquet")
	test = spark.read.parquet("test_set.parquet")

	# Drop unrelated columns
	cols_to_drop = ['is_read', 'is_reviewed']
	train = train.drop(*cols_to_drop)
	val = val.drop(*cols_to_drop)
	test = test.drop(*cols_to_drop)

	# Range of hyper parameters 
	ranks = [15, 20, 25, 30]
	regParams = [0.01, 0.05, 0.1, 0.25, 0.5]
	param_grid = itertools.product(ranks, regParams)

	# Initialize empty values for best parameters
	best_model = MatrixFactorizationModel
	best_rank = 0
	best_lambda = -1.0
	best_rmse = 5
	rmse_value = []
	r = 1

	# Start hyper parameter tuning
	for i in param_grid:
		print("Run = {0}/{1}".format(r,len(list(param_grid))))
		print('##### Training for (rank, lambda) = {} #####'.format(i))
		als = ALS(rank = i[0], regParam=i[1], userCol="user_id", itemCol="book_id", ratingCol="rating",\
				nonnegative=True, coldStartStrategy="drop")
		evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating',predictionCol='prediction')
		model = als.fit(train)
		preds = model.transform(val).sort(desc("rating"))
		rmse = evaluator.evaluate(preds)
		rmse_value.append(rmse)
		rmse_value

		if (rmse <= min(rmse_value)):
			best_rmse = rmse
			best_model = model
			best_rank = best_model.rank
			best_lambda = best_model._java_obj.parent().getRegParam()
		r += 1

	# Results for the best performing model (lowest RMSE) on validation data
	model_path = "hdfs:/user/ago265/best_model"
	best_model.save(model_path)
	print("Best Rank = ", best_rank)
	print("Best Lambda = ", best_lambda)
	print("Best RMSE for validation data = ", best_rmse)
	# Results on test data
	print("Transforming test data")
	preds_test = best_model.transform(test).sort(desc("rating"))
	rmse_test = evaluator.evaluate(preds_test)
	print("RMSE for test data = ", rmse_test)

if __name__ == "__main__":

	memory = "16g"
	spark = SparkSession.builder \
				.appName('als') \
				.master('yarn') \
				.config('spark.executor.memory', memory) \
				.config('spark.driver.memory', memory) \
				.config('spark.executor.memoryOverhead', memory) \
				.config("spark.sql.broadcastTimeout", "36000") \
				.config("spark.storage.memoryFraction","0") \
				.config("spark.memory.offHeap.enabled","true") \
				.config("spark.memory.offHeap.size",memory) \
				.getOrCreate()

	train_model(spark)