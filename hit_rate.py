from itertools import count
import pyspark.sql.functions as F
from pyspark.ml.evaluation import Evaluator
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.ml import Estimator
from pyspark.sql import DataFrame

class HitRate(Evaluator):
    def __init__(self, predictionCol='prediction', labelCol='label', userCol = 'userID', rowNumber = 'rowNumber', k = 10):
        self.predictionCol = predictionCol
        self.labelCol = labelCol
        self.userCol = userCol
        self.rowNumber = rowNumber
        self.k = k
        
    def leaveOneOut(self, dataset):
      windowSpec  = Window.partitionBy(self.userCol).orderBy(F.col(self.labelCol).desc)
      return dataset.withColumn("row_number",row_number().over(windowSpec)).filter(F.col("row_numer") == 1)

    def eval(self, dataframe : DataFrame, estimator : Estimator) -> float:
      totalHit = 0
      for userRow in dataframe.selectExpr("distinct({})".format(self.userCol)).take(100):
        userID = userRow[self.userCol]
        # leave one out
        loo_df = dataframe.filter('{}!={} or {}!={}'.format(self.rowNumber, 1, self.userCol, userID))
        left_out = dataframe.filter('{}={} and {}={}'.format(self.rowNumber, 1, self.userCol, userID))
        model = estimator.fit(loo_df)

        result = model.transform(dataframe)
        result.show()
        windowSpec  = Window.partitionBy(self.userCol).orderBy(F.col(self.predictionCol).desc)
        result = result.withColumn("prediction_row_number", row_number().over(windowSpec))
        isEmpty = result.selectExpr("'{}'={} and '{}'={}".format(self.userCol, left_out[self.userCol], "MovieID", left_out["MovieID"])).rdd.isEmpty()

        if (isEmpty):
          totalHit = totalHit + 1
      print(totalHit)

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