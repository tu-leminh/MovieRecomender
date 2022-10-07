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
        test.unpersist(blocking = True)
        result.unpersist(blocking = True)
      return totalHit / self.repeats

    def _evaluate(self, dataset, gt):
      return

    def isLargerBetter(self):
        return True