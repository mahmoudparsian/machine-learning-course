# TF-IDF

![TF-IDF](./TF-IDF.png)

* What is TF-IDF?
	* TF = term’s frequency
	* IDF = inverse document frequency (measure of how significant that term is in the whole corpus.)

````
    TF-IDF  is an information retrieval technique 
    that weighs a term’s frequency (TF) and its 
    inverse document frequency (IDF). 
    
    Each word or term has its respective TF and IDF 
    score. The product of the TF and IDF scores of
    a term is called the TF-IDF weight of that term.
 
    Put simply, the higher the TF-IDF score (weight), 
    the rarer the term and vice versa.
````

## Basic Definitions
   
N = total number of documents in the corpus

D = set of documents = {d1, ..., dN}

For a term t in a document d, the weight W(t,d) 
of term t in document d is given by:

````
TF-IDF = TF(t,d) * IDF(t)

or

W(t, d) = TF(t,d) * IDF(t)

W(t,d) = TF(t,d) *  log ( N / DF(t) )

or 

W(t,d) = TF(t,d) *  log ( N / (DF(t)+1) )
````

Where:

* `TF(t,d)` is the number of occurrences of t in document d.

* `DF(t)` is the number of documents containing the term t.


## TF-IDF DEFINED
How is TF-IDF calculated? 
The TF (term frequency) of a word is the frequency 
of a word (i.e. number of times it appears) in a 
document. When you know  it, you’re able to see if 
you’re using a term too much or too little.

For example, when a 100-word document contains the 
term  “cat” 12 times, the TF for the word ‘cat’ is

````
TF(cat) = 12/100 = 0.12
TF(dog) = 6/100  = 0.06
````

The IDF (inverse document frequency) of a word 
is the measure of how significant that term is 
in the whole corpus.

Let’s say the term “cat” appears x amount of times 
in a 10,000,000 million document-sized corpus (i.e. 
web). Let’s assume there are 0.3 million documents 
that contain the term “cat”, then the IDF (i.e. 
log {DF}) is given by the total number of documents 
(10,000,000) divided by the number of documents 
containing the term “cat” (300,000).

````
log(of-base-10)

IDF (cat) = log (10,000,000/300,000) = 1.52
IDF (dog) = log (10,000,000/1000,000) = 1.00


W(cat) = TF-IDF(cat) = 0.12 * 1.52 = 0.182
W(dog) = TF-IDF(dog) = 0.06 * 1.00 = 0.06
````


HOW YOU CAN BENEFIT FROM USING TF-IDF
=====================================
Gather words. Write your content. 

Run a TF-IDF report for your words and get their weights. 

* The higher the numerical weight value, the rarer the term. 

* The smaller the weight, the more common the term. 

Compare all the terms with high TF-IDF weights with 
respect to their search volumes on the web. Select 
those with higher search volumes and lower competition. 
Work smart.

A good rule of thumb is, the more your content 
“makes sense” to the user, the more weight it is 
assigned by the search engine. 

With words having a high TF-IDF weight in your content, 
your content will always be among the top search results, 
so you can:

stop worrying about using the stop-words,
successfully hunt words with higher search 
volumes and lower competition, be sure to have 
words that make your content unique and relevant 
to the user, etc.

## Example

````
N = 5

d0 = 'fox jumped fox jumped fox is high low'
d1 = 'fox ran fox ran'
d2 = 'fox jumped high'
d3 = 'fox is red fox is red'
d4 = 'dog is blue dog is blue'
````

### Basic calculations

````
d0: TF-IDF(fox) = ?
TF(fox, d0) = 3 / 8 = 0.375
IDF (fox) = log (5/4) = 0.09
TF-IDF(fox) = 0.375 * 0.09 = 0.03
TF-IDF(fox) = 0.375 * (0.09 +1) = 0.03

d0: TF-IDF(jumped) = ?
TF(jumped, d0) = 2 / 8 = 0.25 
IDF (jumped) = log (5/2) = 0.3979
TF-IDF(fox) = 0.25 * 0.3979 = 0.099
````

### SciKit example

````
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
data = [
    'fox jumped fox jumped fox is high low',
    'fox ran fox ran',
    'fox jumped high',
    'fox is red fox is red',
    'dog is blue dog is blue'
  ]
  
print(data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)
print(vectorizer.get_feature_names())
z = X.toarray()
print(z)

"""
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> data = [
...     'fox jumped fox jumped fox is high low',
...     'fox ran fox ran',
...     'fox jumped high',
...     'fox is red fox is red',
...     'dog is blue dog is blue'
...   ]
>>>
... print(data)
['fox jumped fox jumped fox is high low', 'fox ran fox ran', 'fox jumped high', 'fox is red fox is red', 'dog is blue dog is blue']
>>> vectorizer = CountVectorizer()
>>> X = vectorizer.fit_transform(data)
>>> print(vectorizer.get_feature_names())
['blue', 'dog', 'fox', 'high', 'is', 'jumped', 'low', 'ran', 'red']
>>> z = X.toarray()
>>> print(z)
[[0 0 3 1 1 2 1 0 0]
 [0 0 2 0 0 0 0 2 0]
 [0 0 1 1 0 1 0 0 0]
 [0 0 2 0 2 0 0 0 2]
 [2 2 0 0 2 0 0 0 0]]
>>>
"""


vectorizer1 = TfidfVectorizer(min_df=1)
X1 = vectorizer1.fit_transform(data)
idf = vectorizer1.idf_
print (dict(zip(vectorizer1.get_feature_names(), idf)))

print(X1.toarray())

"""
>>> vectorizer1 = TfidfVectorizer(min_df=1)
>>> X1 = vectorizer1.fit_transform(data)
>>> idf = vectorizer1.idf_
>>> print (dict(zip(vectorizer1.get_feature_names(), idf)))
{'blue': 2.09861228866811, 
 'dog': 2.09861228866811, 
 'fox': 1.1823215567939547, 
 'high': 1.6931471805599454, 
 'is': 1.4054651081081644, 
 'jumped': 1.6931471805599454, 
 'low': 2.09861228866811, 
 'ran': 2.09861228866811, 
 'red': 2.09861228866811
}
>>>
>>> print(X1.toarray())
[[0.         0.         0.61471324 0.29343399 0.24357672 0.58686797 0.36370386 0.         0.        ]
 [0.         0.         0.49084524 0.         0.         0.			0.         0.87124678 0.        ]
 [0.         0.         0.44274009 0.63402729 0.         0.63402729	0.         0.         0.        ]
 [0.         0.         0.42395393 0.         0.5039682  0.			0.         0.         0.75251519]
 [0.63907044 0.63907044 0.         0.         0.42799292 0.			0.         0.         0.        ]]
>>>
"""

vectorizer2 = TfidfVectorizer()
X = vectorizer2.fit_transform(data)
print(vectorizer2.get_feature_names())
X
print("X=", str(X))


>>> vectorizer2 = TfidfVectorizer()
>>> X = vectorizer2.fit_transform(data)
>>> print(vectorizer2.get_feature_names())
['blue', 'dog', 'fox', 'high', 'is', 'jumped', 'low', 'ran', 'red']
>>> X
<5x9 sparse matrix of type '<class 'numpy.float64'>'
	with 16 stored elements in Compressed Sparse Row format>
>>> print("X=", str(X))
X=   
  (0, 6)	0.36370386258015436
  (0, 3)	0.2934339862639287
  (0, 4)	0.24357671557569646
  (0, 5)	0.5868679725278574
  (0, 2)	0.6147132359888916
  (1, 7)	0.8712467800931323
  (1, 2)	0.4908452385195859
  (2, 3)	0.6340272916188496
  (2, 5)	0.6340272916188496
  (2, 2)	0.44274008962926825
  (3, 8)	0.7525151949161979
  (3, 4)	0.5039681962632367
  (3, 2)	0.42395393449691715
  (4, 0)	0.6390704413963749
  (4, 1)	0.6390704413963749
  (4, 4)	0.42799292268317357
````

# References

1. [TF-IDF, wiki](https://en.wikipedia.org/wiki/Tf–idf)

2. [TF-IDF](http://www.tfidf.com)

3. [What is TF-IDF?](https://monkeylearn.com/blog/what-is-tf-idf/)

4. [TF(Term Frequency)-IDF(Inverse Document Frequency) from scratch in python](https://towardsdatascience.com/tf-term-frequency-idf-inverse-document-frequency-from-scratch-in-python-6c2b61b78558)
