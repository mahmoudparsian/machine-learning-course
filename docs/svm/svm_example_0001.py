mparsian@CNSH-OLW-015780 ~/zmp/github/machine-learning-course/docs/LDA (master) $ python3
Python 3.7.2 (v3.7.2:9a3ffc0492, Dec 24 2018, 02:44:43)
[Clang 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from sklearn import svm
>>> X = [[0, 0], [1, 1], [0.5, 0.5], [4, 4], [5, 5], [4.5, 4.5]]
>>> X
[[0, 0], [1, 1], [0.5, 0.5], [4, 4], [5, 5], [4.5, 4.5]]
>>> y = [0, 0, 0, 1, 1, 1]
>>> y
[0, 0, 0, 1, 1, 1]

>>> plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
<matplotlib.collections.PathCollection object at 0x12032dd68>
>>> plt.show()

>>> model = svm.SVC()
>>> model.fit(X, y)
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
>>> model = svm.SVC(gamma='auto')
>>> model.fit(X, y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
>>> model.predict([[2., 2.]])
array([0])
>>> model.predict([[3., 3.]])
array([1])
>>> model.predict([[7, 7]])
array([1])
>>> model.support_vectors_
array([[0., 0.],
       [1., 1.],
       [4., 4.],
       [5., 5.]])
>>> # get indices of support vectors
...
>>> model.support_
array([0, 1, 3, 4], dtype=int32)
>>> # get number of support vectors for each class
...
>>> model.n_support_
array([2, 2], dtype=int32)
>>>
>>>
>>>
>>> print(__doc__)
None
>>>
>>>
>>>
>>>
>>>
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from sklearn import svm
>>> from sklearn.datasets import make_blobs
>>>
>>> # we create 40 separable points
...
>>>
>>> X, y = make_blobs(n_samples=40, centers=2, random_state=6)
>>> X
array([[  6.37734541, -10.61510727],
       [  6.50072722,  -3.82403586],
       [  4.29225906,  -8.99220442],
       [  7.39169472,  -3.1266933 ],
       [  7.64306311, -10.02356892],
       [  8.68185687,  -4.53683537],
       [  5.37042238,  -2.44715237],
       [  9.24223825,  -3.88003098],
       [  5.73005848,  -4.19481136],
       [  7.9683312 ,  -3.23125265],
       [  7.37578372,  -8.7241701 ],
       [  6.95292352,  -8.22624269],
       [  8.21201164,  -1.54781358],
       [  6.85086785,  -9.92422452],
       [  5.64443032,  -8.21045789],
       [ 10.48848359,  -2.75858164],
       [  7.27059007,  -4.84225716],
       [  6.29784608, -10.53468031],
       [  9.42169269,  -2.6476988 ],
       [  8.98426675,  -4.87449712],
       [  6.6008728 ,  -8.07144707],
       [  5.95313618,  -6.82945967],
       [  6.87151089, -10.18071547],
       [  6.26221548,  -8.43925752],
       [  7.97164446,  -3.38236058],
       [  7.67619643,  -2.82620437],
       [  7.92736799,  -9.7615272 ],
       [  5.86311158, -10.19958738],
       [  8.07502382,  -4.25949569],
       [  6.78335342,  -8.09238614],
       [  7.89359985,  -7.41655113],
       [  6.04907774,  -8.76969991],
       [  6.77811308,  -9.80940478],
       [  8.71445065,  -2.41730491],
       [  8.49142837,  -2.54974889],
       [  9.49649411,  -3.7902975 ],
       [  7.52132141,  -2.12266605],
       [  6.3883927 ,  -9.25691447],
       [  7.93333064,  -3.51553205],
       [  6.86866543, -10.02289012]])
>>> y
array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1,
       1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1])
>>> # fit the model, don't regularize for illustration purposes
...
>>> plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
<matplotlib.collections.PathCollection object at 0x12032dd68>
>>> plt.show()

>>> clf = svm.SVC(kernel='linear', C=1000)
>>> clf.fit(X, y)
SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='linear', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
>>> plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
<matplotlib.collections.PathCollection object at 0x11d6740f0>
>>> # plot the decision function
...
>>> ax = plt.gca()
>>> xlim = ax.get_xlim()
>>> ylim = ax.get_ylim()
>>> # create grid to evaluate model
...
>>> xx = np.linspace(xlim[0], xlim[1], 30)
>>> yy = np.linspace(ylim[0], ylim[1], 30)
>>> YY, XX = np.meshgrid(yy, xx)
>>> xy = np.vstack([XX.ravel(), YY.ravel()]).T
>>> Z = clf.decision_function(xy).reshape(XX.shape)
>>>
>>> # plot decision boundary and margins
...
>>> ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
...            linestyles=['--', '-', '--'])
<matplotlib.contour.QuadContourSet object at 0x11d674198>
>>>
>>> # plot support vectors
...
>>> ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
...            linewidth=1, facecolors='none', edgecolors='k')
<matplotlib.collections.PathCollection object at 0x11d674550>
>>>
>>> plt.show()
