import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.sql import DataFrame

from pyspark.ml import Estimator

class HitRate():
    def __init__(self, predictionCol='prediction', labelCol='label', userCol = 'userID', itemCol = 'itemID', rowNumber = 'RowNumber', k = 10):
        self.predictionCol = predictionCol
        self.labelCol = labelCol
        self.userCol = userCol
        self.itemCol = itemCol
        self.rowNumber = rowNumber
        self.k = k
        
    def leaveOneOut(self, dataset):
      windowSpec  = Window.partitionBy(self.userCol).orderBy(F.col(self.labelCol).desc())
      tmp = dataset.withColumn("row_number",row_number().over(windowSpec))
      tmp.cache().count()
      train = tmp.filter(F.col("row_number") != 1)
      train.cache().count()

      test = tmp.filter(F.col("row_number") == 1)
      test.cache().count()
      return [train, test]

    def eval(self, estimator:Estimator, dataframe : DataFrame) -> float:
      totalHit = 0
      user_df = dataframe.select(self.userCol).distinct()
      movie_df = dataframe.select(self.itemCol).distinct()
      full_matrix = user_df.crossJoin(movie_df)

      [train, test] = self.leaveOneOut(dataframe)
      model = estimator.fit(train.dropna())

      windowSpec  = Window.partitionBy(self.userCol).orderBy(F.col(self.predictionCol).desc())
      result = model.transform(full_matrix)
      result = result.withColumn("prediction_row_number", row_number().over(windowSpec))
      result = result.filter(result["prediction_row_number"] <= self.k)
      result.show()
      test.show()
      tmp = result.join(test, (result[self.userCol] == test[self.userCol]) & (result[self.itemCol] == result[self.itemCol]), "inner")
      tmp.show()
      return tmp.count() / user_df.count()