"""
Classification in Spark
The intent of this blog is to demonstrate binary 
classification in pySpark. The various steps involved in 
developing a classification model in pySpark are as follows:

1) Initialize a Spark session

2) Download and read the the dataset

3) Developing initial understanding about the data

4) Handling missing values

5) Scalerizing the features

6) Train test split

7) Imbalance handling

8) Feature selection

9) Performance evaluation
"""

#-----------------------------
#1) Initialize a Spark session
#-----------------------------
# Initializing a Spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder\
                    .appName("diabeties")\
                    .config("spark.some.config.option","some-value")
                    .getOrCreate()


#-----------------------------
#2) Download and read the the dataset
#-----------------------------
spam_path = '/Users/mparsian/zmp/github/machine-learning-course/data/spam.csv'
ham_path = '/Users/mparsian/zmp/github/machine-learning-course/data/ham.csv'

#-----------------------------
#3) Developing initial understanding about the data
#-----------------------------
spam = spark.sparkContext.textFile(spam_path)
ham = spark.sparkContext.textFile(ham_path) 

spam_words = spam.map(lambda email: email.split())
ham_words = ham.map(lambda email: email.split())

#-----------------------------
# 4) convert: Then, we’re hashing each message into 1,000 word buckets. 
#  As you can see, each message is turned into a sparse vector holding 
# bucket numbers and occurrences.
from pyspark.mllib.feature import HashingTF
tf = HashingTF(numFeatures = 1000)
spam_features = tf.transform(spam_words)
ham_features = tf.transform(ham_words)

print(spam_features.take(1))
print(ham_features.take(1))

#-----------------------------
# 5) Scalerizing the features
# The next step is to label our features: 1 for spam, 0 for non-spam. 
# The result is a collected of labeled samples which are ready for use.
from pyspark.mllib.regression import LabeledPoint
spam_samples = spam_features.map(lambda features:LabeledPoint(1, features))
ham_samples = ham_features.map(lambda features:LabeledPoint(0, features))

print(spam_samples.take(1))
print(ham_samples.take(1))
#-----------------------------

#-----------------------------
# 6) Train test split
# Finally, we split the data set 80/20 for training and test 
# and cache both RDDs as we will use them repeatedly.
#-----------------------------
samples = spam_samples.union(ham_samples)
[training_data, test_data] = samples.randomSplit([0.8, 0.2])
training_data.cache()
test_data.cache()
training_data.count()
test_data.count()
#-----------------------------


#-----------------------------
# 7) Performance evaluation
#-----------------------------
def score(model):
   predictions = model.predict(test_data.map(lambda x: x.features))
   labels_and_preds = test_data.map(lambda x: x.label).zip(predictions)
   accuracy = labels_and_preds.filter(lambda x: x[0] == x[1]).count() / float(test_data.count())
   return accuracy

#----------------------
# Algorithms
#----------------------
from pyspark.mllib.classification import LogisticRegressionWithSGD

algo = LogisticRegressionWithSGD()
model = algo.train(training_data)
score(model)

spamExample = tf.transform("You have won $1,000,000. Please fly to Nigeria ASAP. This is urgent".split(" "))
hamExample = tf.transform("Spark is really good at big data processing".split(" "))

print(model.predict(spamExample))
print(model.predict(hamExample))


from pyspark.mllib.classification import LogisticRegressionWithLBFGS

algo = LogisticRegressionWithLBFGS()
model = algo.train(training_data)
score(model)

#### Support Vector Machines
#### What about SVMs, another popular algorithm?
from pyspark.mllib.classification import SVMWithSGD
algo = SVMWithSGD()
model = algo.train(training_data)
score(model)


##### Trees
#####
##### Now let’s try three variants of tree-based classification. 
##### The API is slightly different from previous algos.
from pyspark.mllib.tree import DecisionTree

from pyspark.mllib.tree import GradientBoostedTrees

from  pyspark.mllib.tree import RandomForest

algo = DecisionTree()
model = algo.trainClassifier(training_data,numClasses=2,categoricalFeaturesInfo={})
score(model)


algo = GradientBoostedTrees()
model = algo.trainClassifier(training_data,categoricalFeaturesInfo={},numIterations=10)
score(model)

algo = RandomForest()
model = algo.trainClassifier(training_data,numClasses=2,categoricalFeaturesInfo={},numTrees=16)
score(model)

#### Naive Bayes
#### Last but not least, let’s try the Naives Bayes classifier.
from pyspark.mllib.classification import NaiveBayes
algo = NaiveBayes()
model = algo.train(training_data)
score(model)

