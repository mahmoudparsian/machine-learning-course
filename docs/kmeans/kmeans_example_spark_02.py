"""
An example demonstrating k-means clustering.

Run with:

  bin/spark-submit kmeans_example_spark.py

This example requires NumPy (http://www.numpy.org/).
"""

from __future__ import print_function
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.sql import SparkSession

spark = SparkSession\
   .builder\
   .appName("KMeansExample")\
   .getOrCreate()

# Read data set
sample_kmean_data =  "/Users/mparsian/zmp/github/machine-learning-course/data/sample_kmeans_data.txt"   
dataset = spark.read.format("libsvm").load(sample_kmean_data)
dataset.show(truncate=False)
dataset.printSchema()

# Trains a K-means model.
K = 2
kmeans = KMeans().setK(K).setSeed(1)
model = kmeans.fit(dataset)

# Make predictions
predictions = model.transform(dataset)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
   print(center)

# done
spark.stop()
