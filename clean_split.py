# Create a spark session if running py 
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("YourSessionName").getOrCreate()


from pyspark.sql import *
# Load the data
interactions_og = spark.read.format('csv').option("header","true").load("hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv",schema = 'user_id INT, book_id INT, is_read INT, rating INT, is_reviewed INT')
user_id_map = spark.read.format('csv').option("header","true").load("hdfs:/user/bm106/pub/goodreads/user_id_map.csv")
book_id_map = spark.read.format('csv').option("header","true").load("hdfs:/user/bm106/pub/goodreads/book_id_map.csv")

# Create views
interactions_og.createOrReplaceTempView('interactions_og')
user_id_map.createOrReplaceTempView('user_id_map')
book_id_map.createOrReplaceTempView('book_id_map')

# Clean the data
    # rating = 0
interactions_r = spark.sql('SELECT * FROM interactions_og WHERE rating != 0')
interactions_r.createOrReplaceTempView('interactions_r')
    # interactions < 10
to_keep = spark.sql('SELECT user_id, COUNT(book_id) as C FROM interactions_r GROUP BY user_id HAVING C>10')
to_keep.createOrReplaceTempView('to_keep')
interactions = spark.sql('SELECT i.* FROM interactions_r i INNER JOIN to_keep k ON i.user_id = k.user_id ORDER BY i.user_id')

''' Complete dataset '''
# Split the data: Train = 60%, Validation = 20%, Test = 20%
train_set, val_set, test_set = interactions.randomSplit([0.6, 0.2, 0.2], seed=20)

# Validation split: train = 10%, test = 10%
from pyspark.sql.window import Window
window = Window.partitionBy('user_id').orderBy('book_id') 
import pyspark.sql.functions as F
val_set = (val_set.select("user_id","book_id","is_read","rating","is_reviewed",F.row_number().over(window).alias("row_number")))
#even number
val_train_set = val_set.filter(val_set.row_number % 2 == 0).drop('row_number')
#odd number
val_test_set = val_set.filter(val_set.row_number % 2 != 0).drop('row_number')

# Test split: train = 10%, test = 10%
test_set = (test_set.select("user_id","book_id","is_read","rating","is_reviewed",F.row_number().over(window).alias("row_number")))
#even number
test_train_set = test_set.filter(test_set.row_number % 2 == 0).drop('row_number')
#odd number
test_test_set = test_set.filter(test_set.row_number % 2 != 0).drop('row_number')

# Join all training sets
train_final = train_set.union(val_train_set).union(test_train_set)

# Write out to parquet
train_pq = train_final.write.mode('overwrite').parquet("train_set.parquet")
val_pq = val_test_set.write.mode('overwrite').parquet("val_set.parquet")
test_pq = test_test_set.write.mode('overwrite').parquet("test_set.parquet")



''' Test runs with 1% of data '''
interactions1 = interactions.sample(False, 0.01, seed=0)
# Split the data: Train = 60%, Validation = 20%, Test = 20%
train_set1, val_set1, test_set1 = interactions1.randomSplit([0.6, 0.2, 0.2], seed=20)

# Validation split: train = 10%, test = 10%
val_set1 = (val_set1.select("user_id","book_id","is_read","rating","is_reviewed",F.row_number().over(window).alias("row_number")))
#even number
val_train_set1 = val_set1.filter(val_set1.row_number % 2 == 0).drop('row_number')
#odd number
val_test_set1 = val_set1.filter(val_set1.row_number % 2 != 0).drop('row_number')

# Test split: train = 10%, test = 10%
test_set1 = (test_set1.select("user_id","book_id","is_read","rating","is_reviewed",F.row_number().over(window).alias("row_number")))
#even number
test_train_set1 = test_set1.filter(test_set1.row_number % 2 == 0).drop('row_number')
#odd number
test_test_set1 = test_set1.filter(test_set1.row_number % 2 != 0).drop('row_number')

# Join all training sets
train_final1 = train_set1.union(val_train_set1).union(test_train_set1)

# Write out to parquet
train_pq1 = train_final1.write.mode('overwrite').parquet("train_set1.parquet")
val_pq1 = val_test_set1.write.mode('overwrite').parquet("val_set1.parquet")
test_pq1 = test_test_set1.write.mode('overwrite').parquet("test_set1.parquet")



''' Test runs with 5% of data '''
interactions5 = interactions.sample(False, 0.05, seed=0)
# Split the data: Train = 60%, Validation = 20%, Test = 20%
train_set5, val_set5, test_set5 = interactions5.randomSplit([0.6, 0.2, 0.2], seed=20)

# Validation split: train = 10%, test = 10%
val_set5 = (val_set5.select("user_id","book_id","is_read","rating","is_reviewed",F.row_number().over(window).alias("row_number")))
#even number
val_train_set5 = val_set5.filter(val_set5.row_number % 2 == 0).drop('row_number')
#odd number
val_test_set5 = val_set5.filter(val_set5.row_number % 2 != 0).drop('row_number')

# Test split: train = 10%, test = 10%
test_set5 = (test_set5.select("user_id","book_id","is_read","rating","is_reviewed",F.row_number().over(window).alias("row_number")))
#even number
test_train_set5 = test_set5.filter(test_set5.row_number % 2 == 0).drop('row_number')
#odd number
test_test_set5 = test_set5.filter(test_set5.row_number % 2 != 0).drop('row_number')

# Join all training sets
train_final5 = train_set5.union(val_train_set5).union(test_train_set5)

# Write out to parquet
train_pq5 = train_final5.write.mode('overwrite').parquet("train_set5.parquet")
val_pq5 = val_test_set5.write.mode('overwrite').parquet("val_set5.parquet")
test_pq5 = test_test_set5.write.mode('overwrite').parquet("test_set5.parquet")



''' Test runs with 25% of data '''
interactions25 = interactions.sample(False, 0.25, seed=0)
# Split the data: Train = 60%, Validation = 20%, Test = 20%
train_set25, val_set25, test_set25 = interactions25.randomSplit([0.6, 0.2, 0.2], seed=20)

# Validation split: train = 10%, test = 10%
val_set25 = (val_set25.select("user_id","book_id","is_read","rating","is_reviewed",F.row_number().over(window).alias("row_number")))
#even number
val_train_set25 = val_set25.filter(val_set25.row_number % 2 == 0).drop('row_number')
#odd number
val_test_set25 = val_set25.filter(val_set25.row_number % 2 != 0).drop('row_number')

# Test split: train = 10%, test = 10%
test_set25 = (test_set25.select("user_id","book_id","is_read","rating","is_reviewed",F.row_number().over(window).alias("row_number")))
#even number
test_train_set25 = test_set25.filter(test_set25.row_number % 2 == 0)
test_train_set25 = test_train_set25.drop('row_number')
#odd number
test_test_set25 = test_set25.filter(test_set25.row_number % 2 != 0)
test_test_set25 = test_test_set25.drop('row_number')

train_final25 = train_set25.union(val_train_set25).union(test_train_set25)

# Write out to parquet
train_pq25 = train_final25.write.mode('overwrite').parquet("train_set25.parquet")
val_pq25 = val_test_set25.write.mode('overwrite').parquet("val_set25.parquet")
test_pq25 = test_test_set25.write.mode('overwrite').parquet("test_set25.parquet")