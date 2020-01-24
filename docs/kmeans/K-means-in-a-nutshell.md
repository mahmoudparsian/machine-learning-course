# K-means in a nutshell


<img src="./k-means2.jpg"
     alt="k-means2.jpg"
     style="float: left; margin-right: 10px;" 
/>

## What is K-means clustering?

````
  K-means  clustering  is  a  method of  
  vector quantization,  originally from 
  signal processing, that is popular for 
  cluster analysis in data mining

  The algorithm always converges (by-definition) 
  but  not  necessarily  to  global  optimum.
````

## How does K-means work?

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


## K-means Algorithm:

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


## K-Means Clustering Algorithm Revised: Detailed

Given k the k-means performs following algorithm the repeatedly:

````
Step-1: Partition objects into k nonempty subsets

Step-2: Compute the centroids of the clusters in 
        the current partition (the centroid is the 
        center, i.e., mean point, of the cluster)

Step-3: Assign each object to the cluster with the 
        nearest centroid

Step-4: Stop when no more new assignments. 
        Otherwise go back to Step 2

````
