# K-nearest neighbors (KNN)

<img src="./knn_example_01.png" width="500" height="300" alt="Machine Learning">
<img src="./knn_example_02.png" width="500" height="300" alt="Machine Learning">

* The k-nearest neighbors (KNN) algorithm is a simple, 
  supervised machine learning algorithm that can be used 
  to solve both 
	* classification problems, and 
	* regression problems. 
  
* KNN algorithm is easy to implement and understand, 

* KNN has a major drawback of becoming significantly slows 
  as the size of that data in use grows.

* In pattern recognition, the k-nearest neighbors algorithm 
  (k-NN) is a non-parametric method used for classification 
  and regression. In both cases, the input consists of the 
  k closest training examples in the feature space.

* Non-parametric method refers to a type of statistic that 
  does not require that the population being analyzed meet 
  certain assumptions, or parameters.

* KNN is a Lazy binding
	* The kNN algorithm belongs to a family of instance-based, competitive learning and lazy learning algorithms.
	* Lazy learning refers to the fact that the algorithm does not build a model until the time a prediction is required. 
	* It is lazy because it only does work at the last second. 
  
------- 
## KNN Algorithm

Steps for finding KNN:

````
1. Determine the value of K = number of 
   nearest neighbors to be considered.
   
   K = 1
   K = 3
   K = 5

2. Calculate the distance (Euclidean is the most 
   popular implementation to work by hand) between 
   the query instance and all the training samples
   
3. Sort the distance and determine nearest neighbors 
   based on the K-th minimum distance

4. Gather the category/class labels of the 
   k nearest neighbors.

5. Use simple majority of the category of 
   nearest neighbors as the prediction 
   label of the query instance
````
-------

<img src="./knn_example_03.png" width="800" height="500" alt="Machine Learning">

-------
## KNN Examples  
 
1. [KNN Numerical example](https://people.revoledu.com/kardi/tutorial/KNN/KNN_Numerical-example.html)

2. [Machine Learning Basics with the K-Nearest Neighbors Algorithm](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)

3. [Tutorial: K Nearest Neighbors in Python](https://www.dataquest.io/blog/k-nearest-neighbors-in-python/)

4. [Machine Learning — KNN using scikit-learn](https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75)

5. [Build KNN classifier using Python Scikit-learn package](https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn)

6. [Python Scikit-learn: K Nearest Neighbors - Split the iris dataset into 70% train data and 30% test data](https://www.w3resource.com/machine-learning/scikit-learn/iris/python-machine-learning-k-nearest-neighbors-algorithm-exercise-4.php)

7. [KNN for Classification using Scikit-learn](https://www.kaggle.com/amolbhivarkar/knn-for-classification-using-scikit-learn)

8. [Develop k-Nearest Neighbors in Python From Scratch](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)

9. [K-Nearest Neighbors Algorithm in Python and Scikit-Learn](https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/)

10. [k-nearest neighbor algorithm in Python](https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/)

11. [k-Nearest-Neighbor Classifier](https://www.python-course.eu/k_nearest_neighbor_classifier.php)

12. [K-Nearest Neighbors Algorithm Using Python](https://www.edureka.co/blog/k-nearest-neighbors-algorithm/)

13. [Introduction to k-Nearest Neighbors: A powerful Machine Learning Algorithm (with implementation in Python & R)](https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/)

14. [k-nearest neighbor algorithm different from k-means clustering?](https://www.quora.com/How-is-the-k-nearest-neighbor-algorithm-different-from-k-means-clustering)

15. [K Nearest Neighbors - Classification](https://www.saedsayad.com/k_nearest_neighbors.htm)

16. [K-Nearest Neighbors (KNN) Algorithm for Machine Learning](https://medium.com/capital-one-tech/k-nearest-neighbors-knn-algorithm-for-machine-learning-e883219c8f26)


--------

## KNN Videos

1. [How kNN algorithm works: video: 4 mins](https://www.youtube.com/watch?v=UqYde-LULfs)

2. [KNN Algorithm - How KNN Algorithm Works With Example: video: 27 mins](https://www.youtube.com/watch?v=4HKqjENq9OU)

3. [K-Nearest Neighbours](https://www.geeksforgeeks.org/k-nearest-neighbours/)

4. [K-nearest neighbors (KNN) algorithm is a type of supervised ML algorithm](https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_knn_algorithm_finding_nearest_neighbors.htm)

5. [KNN: video 10 mins](https://www.youtube.com/watch?v=s-9Qqpv2hTY)

--------


A Complete Guide to K-Nearest-Neighbors with Applications in Python and R
https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/

