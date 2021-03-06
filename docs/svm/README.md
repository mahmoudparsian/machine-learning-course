# SVM -- Support Vector Machine

------

<img src="./svm_00.jpeg"
     alt="svm_00.jpeg"
     style="float: left; margin-right: 10px;"
/>

-------

<img src="./svm_01.png"
     alt="svm_01.png"
     style="float: left; margin-right: 10px;"
/>

-------

<img src="./svm_03.gif"
     alt="svm_03.gif"
     style="float: left; margin-right: 10px;"
/>

-------
<img src="./svm_04.png"
     alt="svm_04.png"
     style="float: left; margin-right: 10px;"
/>

-------

### 1. Introduced in 1992
* Support Vector Machine (SVM) was first 
  heard in 1992, introduced by Boser, Guyon, 
  and  Vapnik in COLT-92. 

### 2. SVM for Classification
* Support vector machines (SVMs) are a set 
  of related supervised learning methods used 
  for classification and regression. 

### 3. SVM as a generalized linear classifiers
* SVMs belong to a family of generalized linear 
classifiers.

### 4. SVM as supervised ML
* Support Vector Machine (SVM) is a supervised 
  machine learning algorithm which can be used 
  for both classification or regression challenges. 
  However,  it is mostly used in classification 
  problems. 

### 5. SVM as Binary Classification
* Support vector machines (SVMs) are a set of 
  supervised learning methods used for classification, 
  regression and outliers detection.
  

------

<img src="./svm_intro_01.png"
     alt="svm_intro_01.png"
     style="float: left; margin-right: 10px;"
/>

------

<img src="./svm_intro_02_hyperplane.png"
     alt="svm_intro_02_hyperplane.png"
     style="float: left; margin-right: 10px;"
/>

------

### SVM as a non-probabilistic  classifier
* A support vector machine (SVM) is a non-probabilistic 
binary linear classifier. The nonprobabilistic aspect 
is its key strength. This aspect is in contrast with 
probabilistic classifiers such as the Naïve Bayes. 

### SVMs are fast by some features
* SVM separates data across a decision boundary 
(plane) determined by only a small subset of the 
data (feature vectors). The data subset that supports 
the decision boundary are aptly called the support 
vectors. The remaining feature vectors of the dataset 
do not have any influence in determining the position 
of the decision boundary in the feature space. 

### Probabilistic classifiers need all data
* In contrast with SVMs, probabilistic classifiers 
  develop a model that best explains the data by 
  considering all of the data versus just a small 
  subset. Subsequently, probabilistic classifiers 
  likely require more computing resources.

### Can SVM be used for multi class classification?
* Yes. While the other answers are right in that, SVM 
(as originally proposed in Vapnik's paper) is inherently 
binary classifier and that often multiclass SVMs are 
implemented in one vs all fashion.

-----

## Videos:
0. [Support Vector Machine (SVM) Tutorial Learning SVMs from examples](https://blog.statsbot.co/support-vector-machines-tutorial-c1618e635e93)

1. [class -- Support Vector Machines (SVMs): A friendly introduction -- 30 mins](https://www.youtube.com/watch?v=Lpr__X8zuE8)

2. [class -- Support Vector Machines: A Visual Explanation with Sample Python Code -- 22 mins](https://www.youtube.com/watch?v=N1vOgolbjSc)

3. [Support Vector Machines, Clearly Explained, video: 20 minutes](https://www.youtube.com/watch?v=efR1C6CvhmE)

4. [How Support Vector Machine Works, video: 26 minutes](https://www.youtube.com/watch?v=TtKF996oEl8)

5. [Lecture 68 — Support Vector Machines Mathematical Formulation | Stanford: 12 minutes](https://www.youtube.com/watch?v=ax8LxRZCORU)

-----

## SVM Lecture Notes

<!--
0. (images: hyperplane)
SVM: Feature Selection and Kernels
https://towardsdatascience.com/svm-feature-selection-and-kernels-840781cc1a6c
-->

0. [An introduction to Support Vector Machines (SVM)](https://monkeylearn.com/blog/introduction-to-support-vector-machines-svm/)

1. [A Gentle Introduction to Support Vector Machines](https://med.nyu.edu/chibi/sites/default/files/chibi/Final.pdf)
<!-- svm_lecture_notes_final.pdf -->

2. [Support Vector Machines  (CMU)]()
<!--  svm_CMU.ppt -->

3. [SVM Slides from Stanford]
<!-- lecture14-SVMs.ppt -->

4. [An Idiot’s guide to Support vector machines (SVMs)](http://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf)

5. [Introduction to Support Vector Machines by Dr. Raj Bridgelall 9/2/2017, 18 Pages](https://www.ugpti.org/smartse/resources/downloads/support-vector-machines.pdf)

6. [lecture6_svm.pptx]()
<!--  lecture6_svm.pptx -->

7. [Introduction to Support Vector Machines](https://docs.opencv.org/2.4/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html)

------

## SVM Scikit

<!--
Feb. 13: DEMO Iris data set
Support Vector Machines Tutorial – Learn to implement SVM in Python
https://data-flair.training/blogs/svm-support-vector-machine-tutorial/

Example of a Kernel: GOOD
https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html

Different kernels example:
Implementing SVM and Kernel SVM with Python's Scikit-Learn
https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

-->

1. SVM using Scikit-Learn in Python (DEMO)
	* [SVM using Scikit-Learn in Python - BLOG](https://www.learnopencv.com/support-vector-machines-svm/)
	* [SVM using Scikit-Learn in Python - CODE](https://www.learnopencv.com/svm-using-scikit-learn-in-python/)

2. [Support Vector Regression (SVR) using linear and non-linear kernels DEMO: CODE](https://scikit-learn.org/0.18/auto_examples/svm/plot_svm_regression.html)

3. [Using SVM to perform classification on a non-linear dataset DEMO: CODE](https://www.geeksforgeeks.org/ml-using-svm-to-perform-classification-on-a-non-linear-dataset/)

4. [Non-linear SVM](https://github.com/htygithub/machine-learning-python/blob/master/SVM/EX1_Non_linear_SVM.md)

5. [Support Vector Machines with Scikit-learn - breast cancer data](https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python)

6. [Implementing SVM and Kernel SVM with Python's Scikit-Learn](https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/)

7. [Chapter 3.1 : SVM from Scratch in Python](https://medium.com/deep-math-machine-learning-ai/chapter-3-1-svm-from-scratch-in-python-86f93f853dc)

8. [Introduction to Machine Learning Algorithms, SVM model from scratch](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)

<!-- 
1.7 (may be) 
Example of linear and non-linear models
https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_svm_non_linear.html

1.8 (may be)
SCIKIT-LEARN : SUPPORT VECTOR MACHINES (SVM) II
https://www.bogotobogo.com/python/scikit-learn/scikit_machine_learning_Support_Vector_Machines_SVM_2.php


2. Classifying data using Support Vector Machines(SVMs) in Python
https://www.geeksforgeeks.org/classifying-data-using-support-vector-machinessvms-in-python/

Support Vector Machines with Scikit-learn
https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

In-Depth: Support Vector Machines
https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html


Support Vector Machine - Classification (SVM)
https://www.saedsayad.com/support_vector_machine.htm

Understanding Support Vector Machine algorithm from examples (along with code)
https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/

Scikit User Manual
1.4. Support Vector Machines
https://scikit-learn.org/stable/modules/svm.html#svm-classification


Mathematical: Support Vector Machine Python Example
https://towardsdatascience.com/support-vector-machine-python-example-d67d9b63f1c8


Linear SVC Machine learning SVM example with Python
https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/


SCIKIT-LEARN : SUPPORT VECTOR MACHINES (SVM)
https://www.bogotobogo.com/python/scikit-learn/scikit_machine_learning_Support_Vector_Machines_SVM.php
-->

-----

## Spark-ML

1. [PySpark Machine Learning Demo](http://www.bdxconsult.com/demo/PySpark_SVM_demo.pdf)

2. [UCLA SVM](http://web.cs.ucla.edu/~mtgarip/linear.html)


https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
In-Depth: Support Vector Machines

-----

## Scikit Examples for SVM

````
1. Support Vector Machines for dummies; A Simple Explanation
https://blog.aylien.com/support-vector-machines-for-dummies-a-simple-explanation/

2. Linear SVC Machine learning SVM example with Python
https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/
        
3. Support Vector Machines with Scikit-learn
https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

4. Implementing SVM and Kernel SVM with Python's Scikit-Learn
https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

5. Support Vector Machines(SVM) in a Nutshell
https://medium.com/datadriveninvestor/getting-started-with-svm-551fec2589d5

6. In-Depth: Support Vector Machines
https://share.cocalc.com/share/8b892baf91f98d0cf6172b872c8ad6694d0f7204/PythonDataScienceHandbook/notebooks/05.07-Support-Vector-Machines.ipynb

7. Understanding Support Vector Machine(SVM) algorithm from examples (along with code)
https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/

8. Support Vector Machine Python Example [DETAILED]
https://towardsdatascience.com/support-vector-machine-python-example-d67d9b63f1c8

9. Choosing C Hyperparameter for SVM Classifiers: Examples with Scikit-Learn
https://queirozf.com/entries/choosing-c-hyperparameter-for-svm-classifiers-examples-with-scikit-learn

````