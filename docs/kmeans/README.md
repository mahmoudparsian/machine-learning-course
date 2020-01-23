# K-meas Clustering


#### What is K-means clustering?

````
  K-means  clustering  is  a  method of  
  vector quantization,  originally from 
  signal processing, that is popular for 
  cluster analysis in data mining

  The algorithm always converges (by-definition) 
  but  not  necessarily  to  global  optimum.
````

#### How does K-means work?

````
  The K-means algorithm starts by randomly 
  choosing a centroid value for each cluster. 
  After that the algorithm iteratively performs 
  three steps: 

  Given a set of data points (x1, x2,..., xn),
  K-means clustering aims to partition the n 
  data points into k (<=n) 
  sets S = {S1, S2, ..., Sk} 
  so as to minimize the within-cluster 
  sum of squares:
  
                k                    2
      arg min SUM  (SUM   || x -Mi ||   )
            i=1   X in Si

  Where Mi is the the mean of points in Si
````


#### K-means Algorithm:

````
   The most commonly used algorithm is Standard 
   algorithm. Standard algorithm begins by assigning 
   k random points in the domain as the mean of each 
   cluster and then it iterates  the following  two 
   steps until it reaches the convergence:

   Step-1:   Find the Euclidean distance between 
             each data instance and centroids of 
             all the clusters; 

   Step-2:   Assign the data instances to the 
             cluster of the centroid with nearest 
             distance; 

   Step-3:   Calculate new centroid values based 
             on the mean values of the coordinates 
             of all the data instances from the 
             corresponding cluster.
````


#### K-means Definition

* [Definition & Math Formula -- wiki](https://en.wikipedia.org/wiki/K-means_clustering)


#### Example of Kmeans

1. [Example of Kmeans Algorithm:](mapreduce_algorithms_for_big_data_analysis_by_kyuseok_shim_KMEANS.pdf)

2. [K-means video: 9 minutes](https://www.youtube.com/watch?v=4b5d3muPQmA)

3. [K-Means Clustering with Scikit-Learn by Example](https://stackabuse.com/k-means-clustering-with-scikit-learn/)

4. [Introduction with image: Clustering using K-means algorithm](https://towardsdatascience.com/clustering-using-k-means-algorithm-81da00f156f6)

5. [K-means Clustering Algorithm: Explained](http://dni-institute.in/blogs/k-means-clustering-algorithm-explained/)

6. [K-means: Step-By-Step Example](http://mnemstudio.org/clustering-k-means-example-1.htm)

7. [Numerical Example of Kmeans clustering](https://people.revoledu.com/kardi/tutorial/kMean/NumericalExample.htm)

8. [Numerical Example of Kmeans clustering](https://www.saedsayad.com/clustering_kmeans.htm)

9. [Step by Step KMeans Explained in Detail](https://www.kaggle.com/shrutimechlearn/step-by-step-kmeans-explained-in-detail)

10. [Demonstration of k-means assumptions](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py)
	* plot_kmeans_assumptions.ipynb

11. [K-means Clustering: Iris data](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html#sphx-glr-auto-examples-cluster-plot-cluster-iris-py)

12. [K-Means Clustering in Python -- detailed](https://mubaris.com/posts/kmeans-clustering/)


#### K-means Spark

1. [Running KMeans clustering on Spark (CLASS Presentation: GOOD)](https://rsandstroem.github.io/sparkkmeans.html)

2. [UCLA Tutorial, K-means Clustering: GOOD WORKING EXAMPLE](http://web.cs.ucla.edu/~zhoudiyu/tutorial/)

3. [Clustering K-means](https://runawayhorse001.github.io/LearningApacheSpark/clustering.html)

4. [In Depth: k-means Clustering: excerpt from the Python Data Science Handbook - docs](https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html)

5. [In Depth: M-means Clustering: excerpt from the Python Data Science Handbook - code](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.11-K-Means.ipynb)
