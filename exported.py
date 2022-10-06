# %%
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import *

from pyspark.sql import functions as F
import pandas as pd

#For windows user only
import os 
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# %%
spark = SparkSession.builder.master("local[*]") \
                    .config('spark.ui.showConsoleProgress', 'false')\
                    .appName('MovieRecomender') \
                    .getOrCreate()
print(spark.sparkContext)
print("Spark App Name : "+ spark.sparkContext.appName)

# %%
schema =             StructType([
                    StructField('UserID', LongType(), True),
                     StructField('MovieID', LongType(), True),
                     StructField('Rating', IntegerType(), True),
                     StructField('Timestamp', LongType(), True),
                     ])

# %%
df = spark.read.option("sep", "::").schema(schema).csv("data/ratings.dat")
df.na.drop()
df = df.toDF(*["UserID", "MovieID", "Rating", "Timestamp"])
df.createOrReplaceTempView("dataset");
df = df.cache()
df.count() #force cache

# %%
sql = '''
select 
  A.UserID, A.MovieID, Rating, ROW_NUMBER() OVER (partition by A.UserID order by Rating desc) as RowNumber
from 
  (
    select 
      * 
    from 
      (
        select 
          distinct(UserID) 
        from 
          dataset
      ), 
      (
        select 
          distinct(MovieID) 
        from 
          dataset
      )
  ) as A left outer join dataset as B
  on (A.UserID, A.MovieID) = (B.UserID, B.MovieID)
'''
full_matrix = spark.sql(sql)
# full_matrix = full_matrix.persist()
full_matrix.show()

# %% [markdown]
# Leave one out for each group in full_matrix

# %%
# (train, test) = df.randomSplit([0.8, 0.2])

# %%
als = ALS(userCol="UserID", itemCol="MovieID", ratingCol="Rating", nonnegative = True, implicitPrefs = False,coldStartStrategy="drop")

# %%
# grid_search = ParamGridBuilder().addGrid(als.rank,[50]).addGrid(als.maxIter,[15]).addGrid(als.regParam, [0.05] ).build()
# #thay đổi hyperparams ở đây và chạy lấy kết quả viết báo cáo

# %%
# evaluator = RegressionEvaluator(metricName="rmse", labelCol="Rating", predictionCol="prediction") 

# %%
# cv = CrossValidator(estimator=als, estimatorParamMaps=grid_search, evaluator=evaluator, numFolds=5)

# %%
# spark.sparkContext.setCheckpointDir('checkpoint/')
# cv_fitted=cv.fit(train)

# %%
# print(cv_fitted.bestModel.rank, cv_fitted.bestModel._java_obj.parent().getMaxIter(),cv_fitted.bestModel._java_obj.parent().getRegParam())

# %%
# evaluator.evaluate(cv_fitted.transform(test).na.drop())

# %%
import pyspark.sql.functions as F
from pyspark.ml.evaluation import Evaluator
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.ml import Estimator
from pyspark.sql import DataFrame

class HitRate(Evaluator):
    def __init__(self, predictionCol='prediction', labelCol='label', userCol = 'userID', itemCol = 'itemID', rowNumber = 'RowNumber', k = 10, repeats = 100):
        self.predictionCol = predictionCol
        self.labelCol = labelCol
        self.userCol = userCol
        self.itemCol = itemCol
        self.rowNumber = rowNumber
        self.k = k
        self.repeats = repeats
        
    def leaveOneOut(self, dataset):
      windowSpec  = Window.partitionBy(self.userCol).orderBy(F.col(self.labelCol).desc)
      return dataset.withColumn("row_number",row_number().over(windowSpec)).filter(F.col("row_numer") == 1)

    def eval(self, dataframe : DataFrame, estimator : Estimator) -> float:
      totalHit = 0
      user_df = dataframe.select(self.userCol).distinct()
      movie_df = dataframe.select(self.itemCol).distinct()
      for userRow in user_df.take(self.repeats):
        userID = userRow[self.userCol]
      
        # leave one out
        leave_out_condition = '{}!={} or {}!={}'.format(self.rowNumber, 1, self.userCol, userID)
        keep_one_condition = '{}={} and {}={}'.format(self.rowNumber, 1, self.userCol, userID)
        loo_df = dataframe.filter(leave_out_condition)
        left_out = dataframe.filter(keep_one_condition).first()

        check_condition = "{}={} and {}={} and prediction_row_number <= {}". \
                format(self.userCol, left_out[self.userCol], self.itemCol, left_out[self.itemCol], self.k)

        model = estimator.fit(loo_df.dropna())
        test = movie_df.withColumn(self.userCol, F.lit(userID))
        result = model.transform(test)
        
        windowSpec  = Window.partitionBy(self.userCol).orderBy(F.col(self.predictionCol).desc())
        result = result.withColumn("prediction_row_number", row_number().over(windowSpec))
        isEmpty = result.filter(check_condition).rdd.isEmpty()
        
        print("Left out: {} with rating of {} and ranking of {}".format(left_out[self.itemCol], left_out[self.labelCol], left_out[self.rowNumber]))
        result.filter("{}={} and {}={}". \
                format(self.userCol, left_out[self.userCol], self.itemCol, left_out[self.itemCol], self.k), ).show()

        if not (isEmpty):
          print(userID + " is a hit")
          totalHit = totalHit + 1

        loo_df.unpersist(blocking = True)
        result.unpersist(blocking = True)
      return totalHit / self.repeats

    def _evaluate(self, dataset, gt):
      windowSpec  = Window.partitionBy(self.userCol).orderBy(F.col(self.predictionCol).desc)
      dataset = dataset.withColumn("row_number",row_number().over(windowSpec))

      #count hit
      res = dataset.filter(F.col("row_number") <= self.k).groupBy(self.userCol).agg(F.sum(F.when(dataset[self.labelCol].isNull(), 0).otherwise(1)).alias("count"))
      res.show()
      res = res.agg(F.count(F.col("count") >= 1).alias("hit"))

      #count user
      user_count = dataset.agg(F.countDistinct(self.userCol).alias("total"))

      return res.first()["hit"] / user_count.first()["total"]

    def isLargerBetter(self):
        return True

# %%
hr_evaluator = HitRate(predictionCol='prediction', labelCol='Rating', userCol='UserID', itemCol = "MovieID", repeats=2)
value = hr_evaluator.eval(full_matrix, als)
print("Hit rate is {}".format(value))
# model = als.fit(train)
# predictions = model.transform(test)
# hr_evaluator.evaluate(predictions)


