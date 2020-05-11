# Bayesian Network

* Bayesian networks (BN) and Bayesian classifiers (BC) 
are traditional probabilistic techniques that have 
been successfully used by various machine learning 
methods to help solving a variety of problems in many 
different domains.


* A Bayesian network, Bayes network, belief network, 
decision network, Bayes(ian) model or probabilistic 
directed acyclic graphical model is a probabilistic 
graphical model (a type of statistical model) that 
represents a set of variables and their conditional 
dependencies via a directed acyclic graph (DAG). 

* Bayesian networks, is a probabilistic directed 
acyclic graphical (DAG) model, a probabilistic graphical 
model (a type of statistical model) that represents a 
set of random variables and their conditional dependencies 
via a directed acyclic graph. 

* Bayesian networks are ideal for taking an event that 
occurred and predicting the likelihood that any one of 
several possible known causes was the contributing factor. 
For example, a Bayesian network could represent the 
probabilistic relationships between diseases and symptoms. 
Given symptoms, the network can be used to compute the
probabilities of the presence of various diseases.


* Efficient algorithms can perform inference and learning in 
Bayesian networks. Bayesian networks that model sequences of 
variables (e.g. speech signals or protein sequences) are called 
dynamic Bayesian networks. 

* Generalizations of Bayesian networks that can represent and 
solve decision problems under uncertainty are called influence diagrams.

Simple Bayes Network
https://upload.wikimedia.org/wikipedia/commons/0/0e/SimpleBayesNet.svg

![Simple Bayes Network](https://upload.wikimedia.org/wikipedia/commons/0/0e/SimpleBayesNet.svg)

--------

## Example
A simple Bayesian network with conditional probability tables.

* Two events can cause grass to be wet: an active sprinkler or rain. 

* Rain has a direct effect on the use of the sprinkler (namely that 
when it rains, the sprinkler usually is not active). 

* This situation can be modeled with a Bayesian network (shown to the right). Each variable has two possible values, T (for true) and F (for false).

* The joint probability function is:

````
Pr(G,S,R)=  P(R) * P(S | R) * P(G | S, R)

where 
  G = "Grass wet (true/false)", 
  S = "Sprinkler turned on (true/false)", and 
  R = "Raining (true/false)".
````

## What are Bayesian networks?
* Bayesian networks are a type of Probabilistic Graphical Model 
  that can be used to build models from data and/or expert opinion.
* Bayesian networks are a concise graphical formalism for 
  describing probabilistic models.


They can be used for a wide range of tasks including prediction, 
anomaly detection, diagnostics, automated insight, reasoning, 
time series prediction and decision making under uncertainty. 
Bayesian networks can be used in:

* analytics disciplines, 
* Descriptive analytics, 
* Diagnostic analytics, 
* Predictive analytics 
* Prescriptive analytics.

### Probabilistic
Bayesian networks are probabilistic because they are built 
from probability distributions and also use the laws of 
probability for prediction and anomaly detection, for 
reasoning and diagnostics, decision making under uncertainty 
and time series prediction.

### Graphical
Bayesian networks can be depicted graphically as shown in Figure 2, 
which shows the well known Asia network. Although visualizing the 
structure of a Bayesian network is optional, it is a great way to 
understand a model.



![Simple Bayes Network Graphical](https://www.bayesserver.com/docs/images/asia-animated.gif)

* A Bayesian network is a directed acyclic graph (DAG)
which is made up of Nodes and directed Links between them.

### Nodes
In many Bayesian networks, each node represents a Variable 
such as someone's height, age or gender. A variable might 
be discrete, such as Gender = {Female, Male} or might be 
continuous such as someone's age.


The nodes and links form the structure of the Bayesian network, 
and we call this the structural specification.


### Discrete
A discrete variable is one with a set of mutually exclusive states 
such as Gender = {Female, Male}.

### Continuous
Bayes networks support continuous variables with Conditional 
Linear Gaussian distributions (CLG). This simply means that 
continuous distributions can depend on each other (are 
multivariate) and can also depend on one or more discrete 
variables.

### Links
Links are added between nodes to indicate that one node 
directly influences the other. When a link does not exist 
between two nodes, this does not mean that they are completely 
independent, as they may be connected via other nodes. They 
may however become dependent or independent depending on 
the evidence that is set on other nodes.

### Directed Acyclic Graph (DAG)
A Bayesian network is a type of graph called a 
Directed Acyclic Graph or DAG. A Dag is a graph 
with directed links and one which contains no 
directed cycles.

##### Directed cycles
A directed cycle in a graph is a path starting and 
ending at the same node where the path taken can 
only be along the direction of links.

### Notation
At this point it is useful to introduce some simple 
mathematical notation for variables and probability distributions.

* Variables are represented with upper-case letters (A,B,C) 
and their values with lower-case letters (a,b,c). If A = a 
we say that A has been instantiated.

A set of variables is denoted by a bold upper-case letter (X), 
and a particular instantiation by a bold lower-case letter (x). 
For example if X represents the variables A,B,C then x is the 
instantiation a,b,c. The number of variables in X is denoted |X|. 
The number of possible states of a discrete variable A is denoted |A|.

* The notation pa(X) is used to refer to the parents of X in a graph. 


* We use P(A) to denote the probability of A.

* We use P(A,B) to denote the joint probability of A and B.

* We use P(A | B) to denote the conditional probability of A given B.

### Probability
P(A) is used to denote the probability of A. 
For example if A is discrete with states {True, False} 
then P(A) might equal [0.2, 0.8]. I.e. 20% chance of 
being True, 80% chance of being False.

### Joint probability
A joint probability refers to the probability of more than 
one variable occurring together, such as the probability 
of A and B, denoted P(A,B).

An example joint probability distribution for variables 
Raining ad Windy is shown below. For example, the probability 
of it being windy and not raining is 0.16 (or 16%).

For discrete variables, the joint probability entries sum to one.

### Joint probability
If two variables are independent (i.e. unrelated) then P(A,B) = P(A)P(B).

### Conditional probability
Conditional probability is the probability of a variable (or set of 
variables) given another variable (or set of variables), denoted P(A|B).

For example, the probability of Windy being True, 
given that Raining is True might equal 50%.

This would be denoted `P(Windy = True | Raining = True) = 50%`.

### Marginal probability
A marginal probability is a distribution formed by 
calculating the subset of a larger probability distribution.

If we have a joint distribution P(Raining, Windy) and 
someone asks us what is the probability of it raining, 
we need P(Raining), not P(Raining, Windy). In order to 
calculate P(Raining), we can simply sum up all the values 
for Raining = False, and Raining = True, as shown below.

````
P(Raining, Windy)

Raining | Windy=False| Windy=True| SUM
False   | 0.64       | 0.16      | 0.80
True    | 0.10       | 0.10      | 0.20
---------------------------------|-----
          0.74         0.26      | 1.00
          
P(Raining)
Raining=False | Raining=True
--------------|--------------
     0.80     |   0.20

````     


### Inference
Inference is the process of calculating a probability distribution of 
interest e.g. P(A | B=True), or P(A,B|C, D=True). The terms inference 
and queries are used interchangeably. The following terms are all forms 
of inference will slightly difference semantics.

Prediction - focused around inferring outputs from inputs.
Diagnostics - inferring inputs from outputs.
Supervised anomaly detection - essentially the same as prediction
Unsupervised anomaly detection - inference is used to calculate the P(e) or more commonly log(P(e)).
Decision making under uncertainty - optimization and inference combined. See Decision graphs for more information.

A few examples of inference in practice:

* Given a number of symptoms, which diseases are most likely?
* How likely is it that a component will fail, given the current state of the system?
* Given recent behavior of 2 stocks, how will they behave together for the next 5 time steps?

## Slides, PDF, Powerpoints, Blogs

1. A Tutorial on Bayesian Networks (slides 48) by Weng-Keen Wong
http://www.cs.ucf.edu/~mingjie/ECM6308/rand0.pdf


2. Basics of Bayesian Network
https://towardsdatascience.com/basics-of-bayesian-network-79435e11ae7b

3. Example of Inference from https://www.cs.ubc.ca/~murphyk/Papers/intro_gm.pdf

Introduction to Bayesian Networks
https://towardsdatascience.com/introduction-to-bayesian-networks-81031eeed94e



--------

## References

* https://www.bayesserver.com/docs/introduction/bayesian-networks

* Bayesian network interactive:
https://www.bayesserver.com/examples/networks/asia

* How To Implement Bayesian Networks In Python? – 
Bayesian Networks Explained With Examples
https://www.edureka.co/blog/bayesian-networks/
