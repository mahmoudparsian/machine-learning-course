from __future__ import print_function

from pyspark.sql import SparkSession

from pyspark.mllib.clustering import KMeans
from numpy import array
from math import sqrt

# create a SparkSession object
spark = SparkSession\
   .builder\
   .appName("KMeansExample")\
   .getOrCreate() 

# Create a SparkContext object
sc = spark.sparkContext

# 4 data points (0.0, 0.0), (1.0, 1.0), (9.0, 8.0) (8.0, 9.0)
data = array([0.0,0.0, 1.0,1.0, 9.0,8.0, 8.0,9.0]).reshape(4,2)
print("data=", data)

#Generate K means
rdd = sc.parallelize(data)
print("rdd.collect()=", rdd.collect())

# build a K-means model
K = 2
model = KMeans.train(rdd, K, maxIterations=10,  initializationMode="random")
print("model=", model)

#Print out the cluster of each data point
print (model.predict(array([0.0, 0.0])))
print (model.predict(array([1.0, 1.0])))
print (model.predict(array([9.0, 8.0])))
print (model.predict(array([8.0, 0.0])))
