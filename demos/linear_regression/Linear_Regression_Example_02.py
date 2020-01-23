Linear regression example with Python code and scikit-learn
Now we are going to write our simple Python program that will represent a linear regression and predict a result for one or multiple data.

In our example, we are going to make our code simpler. So we eliminate to create the plotting graph and only focused on creating a program where we will pass data and it will return the predicted value. I am trying to make the program simpler for better and easy understand and focusing only on the calculation to get the predicted values.

First, let’s import linear_model from scikit-learn library:

from sklearn import linear_model
Now take features and labels set to train our program:

features = [[2],[1],[5],[10]]
labels = [27, 11, 75, 155]
After that create our model and fit the label and features to our model:

clf = linear_model.LinearRegression()
clf=clf.fit(features,labels)
In the end, pass data to the model and print the predicted result:

predicted = clf.predict([[8]])
print(predicted)
 

Now see the complete and final code all together:

from sklearn import linear_model
features = [[2],[1],[5],[10]]
labels = [27, 11, 75, 155]
clf = linear_model.LinearRegression()
clf=clf.fit(features,labels)
predicted = clf.predict([[8]])
print(predicted)
In our program, we have take 8 as the data for which we want to get the predicted result. If we run our program than we will able to see the predicted value. The program actually finds the closest line that will fit closely.

If we want, then we can pass multiple features for which we want to get values like this:

predicted = clf.predict([[8], [3], [11]])
We will get predicted values for each feature we provide.

 

I hope you have understood the example of Python linear example.

One response to “Simple Example of Linear Regression With scikit-learn in Python”
 charles owuor says:
July 13, 2019 at 9:49 pm
In your example, y is used as intercept. this could confuse beginners in regression analysis. y is used as an outcome variable in conventional regression analysis. You could use a different constant.
