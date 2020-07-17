import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, desc
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RankingMetrics


def recsys(spark):
    # Load data from parquet
    val = spark.read.parquet("val_set.parquet")
    test = spark.read.parquet("test_set.parquet")
    cols_to_drop = ['is_read', 'is_reviewed']
    test = test.drop(*cols_to_drop)
    val = val.drop(*cols_to_drop)

    # Load model from path
    model_path = "hdfs:/user/ago265/best_model"
    best_model = ALSModel.load(model_path)

    # Compile a list of all the books each user read
    val_users = val.select("user_id").distinct()

    val_books = val.select("user_id", "book_id")\
                                .groupBy("user_id")\
                                .agg(expr('collect_list(book_id) as books'))

    test_users = test.select("user_id").distinct()
    test_books = test.select("user_id", "book_id").groupBy("user_id").agg(expr('collect_list(book_id) as books'))


    # # Recommender System for all users at k=500
    # k = 500
    # print('Making top 500 recommendations for all users')
    # rec = best_model.recommendForAllUsers(k)

    # Recommender System for subset of users at k=10
    k = 10
    print('Making top {} recommendations for a subset of users'.format(k))
    rec = best_model.recommendForUserSubset(test_users, k)
    pred_label = rec.select('user_id','recommendations.book_id')

    # Create an RDD to evaluate with Ranking Metrics
    final_df = pred_label.join(test_books,['user_id'],'inner').select('book_id','books')
    final_rdd = final_df.rdd.map(lambda x: (x.book_id, x.books))
    
    metrics = RankingMetrics(final_rdd)
    result1 = metrics.meanAveragePrecision
    result2 = metrics.precisionAt(k)
    result3 = metrics.ndcgAt(k)
    print("MAP = ", result1)
    print("Precision at k = ", result2)
    print("NDCG at k = ", result3)



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

	recsys(spark)