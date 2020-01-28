from __future__ import print_function

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql import SparkSession

import sys

spark = SparkSession\
    .builder\
    .appName("NaiveBayesExample")\
    .getOrCreate()

# data_path = "sample_libsvm_data.txt"
data_path = sys.argv[1]
print('data_path=', data_path)
#
# Load training data
data = spark.read.format("libsvm").load(data_path)

# Split the data into train and test
splits = data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
model = nb.fit(train)

# select example rows to display.
predictions = model.transform(test)
predictions.show()

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(\
   labelCol="label",\
   predictionCol="prediction",\
   metricName="accuracy")
   
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))

# done
spark.stop()
