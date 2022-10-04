from itertools import count
import pyspark.sql.functions as F
from pyspark.ml.evaluation import Evaluator
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

class HitRate(Evaluator):
    def __init__(self, predictionCol='prediction', labelCol='label', userCol = 'userId', k = 10):
        self.predictionCol = predictionCol
        self.labelCol = labelCol
        self.userCol = userCol
        self.k = k

    def _evaluate(self, dataset):
      windowSpec  = Window.partitionBy(self.userCol).orderBy(self.predictionCol)
      dataset = dataset.withColumn("row_number",row_number().over(windowSpec))

      #count hit
      res = dataset.filter(F.col("row_number") <= 10).groupBy(self.userCol).agg(F.sum(F.when(dataset[self.labelCol].isNull(), 0).otherwise(1)).alias("count"))
      res.show()
      res = res.agg(F.count(F.col("count") >= 1).alias("hit"))

      #count user
      user_count = dataset.agg(F.countDistinct(self.userCol).alias("total"))

      return res.first()["hit"] / user_count.first()["total"]

    def isLargerBetter(self):
        return True