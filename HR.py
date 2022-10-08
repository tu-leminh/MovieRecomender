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
      train = tmp.filter(F.col("row_number") != 1)
      test = tmp.filter(F.col("row_number") == 1)

      return [train, test]

    def eval( self, estimator:Estimator, dataframe, full_matrix, \
              item_count, user_count) -> float:
      [train, test] = self.leaveOneOut(dataframe)
      model = estimator.fit(train)

      windowSpec  = Window.partitionBy(self.userCol).orderBy(F.col(self.predictionCol).desc())
    
      result = model.transform(full_matrix)
      result = result.repartition(200, self.userCol, self.itemCol)
      test = test.repartition(200, self.userCol, self.itemCol)
      
      
      result = result.withColumn("prediction_row_number", row_number().over(windowSpec))      
      tmp = result.join(test, (result[self.userCol] == test[self.userCol]) & 
                              (result[self.itemCol] == test[self.itemCol]), "inner") \
                  .filter(result["prediction_row_number"] <= self.k)
      return tmp.count() / user_count