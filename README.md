# Logistic-Regression-from-scrarch

#This portion of the coding till the end of second distribution plot is done by Sai Krishna Lakshminarayanan (18230229)Data Analytics

#Importing the packages required for executing the algorithm

#install the missing packages if any is present

import statistics

import random

import numpy as num

from subprocess import check_output

import pandas as panda

import matplotlib.pyplot as plot

from scipy import optimize as op

from sklearn.metrics import confusion_matrix

import csv

import seaborn as sea

from sklearn.cross_validation import train_test_split

#make sure owls.csv is present in the working directory

owls = panda.read_csv('owls.csv')# obtaining the dataset from the local disk

#Data visualisation of the owl data to get some understanding

#Distribution plot of body length

sea.distplot(owls["body-length"])

plot.savefig('body-lengthdistplot.png')#storing the output locally as a png file

#Distribution plot of body width

sea.distplot(owls["body-width"])

plot.savefig('body-widthdistplot.png')

#This portion of the coding till the end of fourth distribution plot is done by Surya Balakrishnan Ramakrishnan (18231072)Data Analytics

#Distribution plot of wing length

sea.distplot(owls["wing-length"])

plot.savefig('wing-lengthdistplot.png')

#Distribution plot of wing width

sea.distplot(owls["wing-width"])

plot.savefig('wing-widthdistplot.png')

#The first two box plots are done by Sai Krishna

#Box plot of the owl types with respect towards the body length

sea.boxplot(x="types", y="body-length", data=owls)

plot.savefig('body-lengthboxplot.png')

#Box plot of the owl types with respect towards the wing length

sea.boxplot(x="types", y="wing-length", data=owls)

plot.savefig('wing-lengthboxplot.png')

#The last two box plots are done by Surya

#Box plot of the owl types with respect towards the body width

sea.boxplot(x="types", y="body-width", data=owls)

plot.savefig('body-widthboxplot.png')

#Box plot of the owl types with respect towards the wing width

sea.boxplot(x="types", y="wing-width", data=owls)

plot.savefig('wing-widthboxplot.png')

#first scatter plot is done by Sai Krishna

#scatter plot of the owl types with respect to body length and width

sea.FacetGrid(owls, hue="types", height=5).map(plot.scatter, "body-length", "body-width")

plot.legend(loc='center right');

plot.savefig('body-lengthVsbody-width.png')

#second scatter plot is done by Surya

#scatter plot of the owl types with respect to wing length and width

sea.FacetGrid(owls, hue="types",height=5).map(plot.scatter, "wing-length", "wing-width")

plot.legend(loc='lower right');

plot.savefig('wing-lengthVswing-width.png')

#This part of the code till the start of regularised cost function is done by Sai Krishna

#Data preprocessing to perform the regression later

types = ['LongEaredOwl', 'SnowyOwl', 'BarnOwl']#the three different types of owls present

e = owls.shape[0] #Total count of the owl data set

f = 4 #No of features it has other than the target variable

g = 3#Number of classes present in the target variable types which denotes the three types of owl

x = num.ones((e,f + 1))

y = num.array((e,1))

x[:,1] = owls['body-length'].values# giving the body length values

x[:,2] = owls['wing-length'].values # giving the wing length values

x[:,3] = owls['body-width'].values# giving the body width values

x[:,4] = owls['wing-width'].values# giving the wing width values

y = owls['types'].values #labels which for which the classification is to be performed

a=10

sum=0

acc=[]

err=[]

for b in range(a): # to perform 10 iteration of the prediction and accuracy

for j in range(f):

X[:, j] = (X[:, j] - X[:,j].mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = b*7)#randomly splitting the data

def sigmoid(z):# to perform the sigmoid operation

return 1.0 / (1 + num.exp(-z))

#from this till the start of accuracy is done by Surya

def regularisedcostfunction(t, X, y, l = 0.1): #Regularized cost function

e = len(y)

h = sigmoid(X.dot(t))

reg = (l/(2 * e)) * num.sum(t**2)

#regularises the cost function by adding proper weights inorder to avoid over or less values

return (1 / e) * (-y.T.dot(num.log(h)) - (1 - y).T.dot(num.log(1 - h))) + reg

def regularisedgradientfunction(t, X, y, l = 0.1): #Regularized gradient function

e, f = X.shape

t = t.reshape((f, 1))

y = y.reshape((e, 1))

h = sigmoid(X.dot(t))

reg = l * t /e

#obtaining the local minimum and minimizing the cost function

return ((1 / e) * X.T.dot(h - y)) + reg

def logreg(X, y, t):

result = op.minimize(fun = regularisedcostfunction, x0 = t, args = (X, y),

method = 'TNC', jac = regularisedgradientfunction)

return result.x

all_t = num.zeros((g, f + 1))

#One vs all

i = 0

for owl in types:

#set the labels in 0 and 1

tmp_y = num.array(y_train == owl, dtype = int)

optTheta = logreg(X_train, tmp_y, num.zeros((f + 1,1)))

all_t[i] = optTheta

i = i+1

#Predictions

P = sigmoid(X_test.dot(all_t.T)) #probability for each owl

p = [types[num.argmax(P[i, :])] for i in range(X_test.shape[0])]



#From this part till the end coding is done by Sai Krishna

count = len(["ok" for ix, label in enumerate(y_test) if label == p[ix]])

c=(float(count) / len(y_test))*100#accuracy value

acc.append(c)

error=1-(c/100)#error value

err.append(error)

print("Test Accuracy of",b+1,"scenario is", c, '%',"\n")# Test accuracy in each iteration

print("Error possibility of",b+1,"scenario is",error,"\n")# erro…

confusionmatrix = confusion_matrix(y_test, p, labels = types) #Regularized gradient function

sea.heatmap(confusionmatrix, annot = True, xticklabels = types, yticklabels = types);#generating heat map along for the matrix

plot.savefig('confusionmatrix.png')

iteration=[]

testtotal=[]

for i in range(a):

iteration.append(i+1)#assigning the iteration values

testtotal.append(y_test.shape[0])#assigning the test total value

final_table = panda.DataFrame(# creating a data set to store the values of iteration,test total,accuracy and error

{

'Number of Iteration ':iteration,

'Test Data':testtotal,

'Accuracy %': acc,

'Error=(1-accuracy)': err,

})

print(final_table)#desired output

final_table.to_csv('finaltable.csv', encoding='utf-8', index=False)#storing the end result as a csv
