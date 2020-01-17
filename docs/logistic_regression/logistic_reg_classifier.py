from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import *
from os import listdir
from os.path import isfile, join
from pyspark.sql.types import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.master("local").appName("Email_Classifier").config("spark.some.config.option", "some-value").getOrCreate() #create a spark session

spampath = './users/rkadam/enron1/spam/'  #add directory path to spam folder
hampath = './users/rkadam/enron1/ham/'    #add directory path to ham folder

old_df = spark.createDataFrame([('', 1.0)], ['text','label'], StringType()) 

for f1 in listdir(spampath):       			#open directory for every email
    p1 = open(spampath+f1, 'r')    			#create a temporary dataframe and append a label '1.0' to ever spam email
    temp1 = spark.createDataFrame([(p1.read(), 1.0)])
    new_df = old_df.unionAll(temp1)                     #make a new dataframe appending all the values from temporary to old
    old_df = new_df
    p1.close()
	
for f2 in listdir(hampath):
    p2 = open(hampath+f2, 'r')				#create a temporary dataframe and append a label '0.0' to ever ham emai
    temp2 = spark.createDataFrame([(p2.read(), 0.0)], ['text', 'label'])
    new_df = old_df.unionAll(temp2)
    old_df = new_df
    p2.close()

final = old_df

(training, test) = final.randomSplit([0.8, 0.2]) 	#split the dataframe into training and test

# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")   	#now create a tokenizer for getting word of the email
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")		#create a feature for every word
lr = LogisticRegression(maxIter=10, regParam=0.001)		#make a logistic regression model
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# Fit the pipeline to training documents.
model = pipeline.fit(training)					#fit the model based on training data

# Make predictions on test documents and print columns of interest.
prediction = model.transform(test)				#make predictions on test data based on model

prediction.show(1)						#show the columns of interest

#Accuracy calculation

evaluator1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator1.evaluate(prediction)

print("Accuracy = %g " % (accuracy))	

#Precision calculation

evaluator2 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedPrecision")

Precision = evaluator2.evaluate(prediction)

print("Precision = %g " % (Precision))

#Recall calculation

evaluator3 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedRecall")

Recall = evaluator3.evaluate(prediction)

print("Recall = %g " % (Recall))
