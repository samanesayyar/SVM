# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 21:41:28 2020


@author: Samane
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report

def sonar_dataset(data):

    # Split train and test and cross validation-60-20-20 %
    array = data.values
    features = array[:,0:60].astype(float)
    labels = array[:,60]

    training_inputs,testing_inputs,training_classes,testing_classes = train_test_split(features,labels,test_size = 0.2,shuffle=True,random_state = 7)
    training_inputs,cross_val_inputs,training_classes,cross_val_classes = train_test_split(training_inputs,training_classes,test_size = 0.25,shuffle=True,random_state = 7)
    return training_inputs, training_classes,cross_val_inputs,cross_val_classes, testing_inputs, testing_classes

# Load dataset
url = 'sonar.all-data.csv'
sonar_data = pd.read_csv(url, header=None)
training_inputs, training_classes,cross_val_inputs,cross_val_classes, testing_inputs, testing_classes=sonar_dataset(sonar_data)

print("Sonar Mines & Rocks Dataset Shape",sonar_data.shape)
print("shape of trainSet",training_inputs.shape)
print("shape of Cross_validationSet",cross_val_inputs.shape)
print("shape of testSet",testing_inputs.shape)
# Define Kernels and Parameters
kernel_list=['linear','poly','rbf','sigmoid']
c_list=[0.1,1,10,100,500,1000]


linear_score_train=[]
linear_score_val=[]
linear_score_test=[]

polynomial_score_train=[]
polynomial_score_val=[]
polynomial_score_test=[]

RBF_score_train=[]
RBF_score_val=[]
RBF_score_test=[]

sigmoid_score_train=[]
sigmoid_score_val=[]
sigmoid_score_test=[]
# train models based on kernels

for C in c_list:
    clf = svm.SVC(C=C, kernel='linear')
    clf.fit(training_inputs, training_classes)
    score_train=clf.score(training_inputs,training_classes)
    linear_score_train.append(score_train)
    score_cros_val=clf.score(cross_val_inputs,cross_val_classes)
    linear_score_val.append(score_cros_val)
    score_test=clf.score(testing_inputs,testing_classes)
    linear_score_test.append(score_test)
    print("Accuracy of C=",C," on trainSet with linear kernel: ",round(score_train*100,2))
    print("Accuracy of C=",C," on Cross_validationSet with linear kernel: ",round(score_cros_val*100,2))
    print("Accuracy of C=",C," on testSet with linear kernel: ",round(score_test*100,2))
    
for C in c_list:
    clf = svm.SVC(C=C, kernel='poly')
    clf.fit(training_inputs, training_classes)
    score_train=clf.score(training_inputs,training_classes)
    polynomial_score_train.append(score_train)
    score_cros_val=clf.score(cross_val_inputs,cross_val_classes)
    polynomial_score_val.append(score_cros_val)
    score_test=clf.score(testing_inputs,testing_classes)
    polynomial_score_test.append(score_test)
    print("Accuracy of C=",C," on trainSet with polynomial kernel: ",round(score_train*100,2))
    print("Accuracy of C=",C," on Cross_validationSet with polynomial kernel: ",round(score_cros_val*100,2))
    print("Accuracy of C=",C," on testSet with polynomial kernel: ",round(score_test*100,2))
    
for C in c_list:
    clf = svm.SVC(C=C, kernel='rbf')
    clf.fit(training_inputs, training_classes)
    score_train=clf.score(training_inputs,training_classes)
    RBF_score_train.append(score_train)
    score_cros_val=clf.score(cross_val_inputs,cross_val_classes)
    RBF_score_val.append(score_cros_val)
    score_test=clf.score(testing_inputs,testing_classes)
    RBF_score_test.append(score_test)
    print("Accuracy of C=",C," on trainSet with rbf kernel: ",round(score_train*100,2))
    print("Accuracy of C=",C," on Cross_validationSet with rbf kernel: ",round(score_cros_val*100,2))
    print("Accuracy of C=",C," on testSet with rbf kernel: ",round(score_test*100,2))
    
for C in c_list:
    clf = svm.SVC(C=C, kernel='sigmoid')
    clf.fit(training_inputs, training_classes)
    score_train=clf.score(training_inputs,training_classes)
    sigmoid_score_train.append(score_train)
    score_cros_val=clf.score(cross_val_inputs,cross_val_classes)
    sigmoid_score_val.append(score_cros_val)
    score_test=clf.score(testing_inputs,testing_classes)
    sigmoid_score_test.append(score_test)
    print("Accuracy of C=",C," on trainSet with sigmoid kernel: ",round(score_train*100,2))
    print("Accuracy of C=",C," on Cross_validationSet with sigmoid kernel: ",round(score_cros_val*100,2))
    print("Accuracy of C=",C," on testSet with sigmoid kernel: ",round(score_test*100,2))



# train best model with best parameters
#C=100

#clf = svm.SVC(C=100, kernel='rbf')
#clf.fit(training_inputs, training_classes)
#score_train=clf.score(training_inputs,training_classes)
#sigmoid_score_train.append(score_train)
#score_cros_val=clf.score(cross_val_inputs,cross_val_classes)
#sigmoid_score_val.append(score_cros_val)
#score_test=clf.score(testing_inputs,testing_classes)
#sigmoid_score_test.append(score_test)

#print("Accuracy of C=",C," on trainSet with sigmoid kernel: ",round(score_train*100,2))
#print("Accuracy of C=",C," on trainSet with sigmoid kernel: ",round(score_cros_val*100,2))
#print("Accuracy of C=",C," on trainSet with sigmoid kernel: ",round(score_test*100,2))
#y_pred = clf.predict(training_inputs)
#print("Classification  report of train data",classification_report(training_classes, y_pred))
#y_pred = clf.predict(cross_val_inputs)
#print("Classification  report of cross_validation data",classification_report(cross_val_classes, y_pred))
#y_pred = clf.predict(testing_inputs)
#print("Classification  report of test data",classification_report(testing_classes, y_pred))
#print('w = ', clf._get_coef())
#print('b = ',clf.intercept_)
## support vectors & bias
#print('Indices of support vectors = ', clf.support_)
#print('Support vectors = ', clf.support_vectors_)
#print('Number of support vectors for each class = ', clf.n_support_)
#print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))    



linear_score_train=np.asarray(linear_score_train).reshape((6,1))
polynomial_score_train=np.asarray(polynomial_score_train).reshape((6,1))
RBF_score_train=np.asarray(RBF_score_train).reshape((6,1))
sigmoid_score_train=np.asarray(sigmoid_score_train).reshape((6,1))
str1="Score on train data"
plt.figure()
plt.title(str1)
plt.plot(c_list,linear_score_train,'r',label="linear")
plt.plot(c_list,polynomial_score_train,'b',label="polynomial")
plt.plot(c_list,RBF_score_train,'g',label="RBF")
plt.plot(c_list,sigmoid_score_train,'y',label="sigmoid")
plt.xlabel('C Value')
plt.ylabel("Score")
plt.legend(loc='best')
   
   
linear_score_val=np.asarray(linear_score_val).reshape((6,1))
polynomial_score_val=np.asarray(polynomial_score_val).reshape((6,1))
RBF_score_val=np.asarray(RBF_score_val).reshape((6,1))
sigmoid_score_val=np.asarray(sigmoid_score_val).reshape((6,1))
str2=str1="Score on cros_validation data"
plt.figure()
plt.title(str2)
plt.plot(c_list,linear_score_val,'r',label="linear")
plt.plot(c_list,polynomial_score_val,'b',label="polynomial")
plt.plot(c_list,RBF_score_val,'g',label="RBF")
plt.plot(c_list,sigmoid_score_val,'y',label="sigmoid")
plt.xlabel('C Value')
plt.ylabel("Score")
plt.legend(loc='best')
    
linear_score_test=np.asarray(linear_score_test).reshape((6,1))
polynomial_score_test=np.asarray(polynomial_score_test).reshape((6,1))
RBF_score_test=np.asarray(RBF_score_test).reshape((6,1))
sigmoid_score_test=np.asarray(sigmoid_score_test).reshape((6,1))
str3="Score on test data"
plt.figure()
plt.title(str3)
plt.plot(c_list,linear_score_test,'r',label="linear")
plt.plot(c_list,polynomial_score_test,'b',label="polynomial")
plt.plot(c_list,RBF_score_test,'g',label="RBF")
plt.plot(c_list,sigmoid_score_test,'y',label="sigmoid")
plt.xlabel('C Value')
plt.ylabel("Score")
plt.legend(loc='best')



linear_score_train=np.asarray(linear_score_train).reshape((6,1))
polynomial_score_train=np.asarray(polynomial_score_train).reshape((6,1))
RBF_score_train=np.asarray(RBF_score_train).reshape((6,1))
sigmoid_score_train=np.asarray(sigmoid_score_train).reshape((6,1))

linear_score_val=np.asarray(linear_score_val).reshape((6,1))
polynomial_score_val=np.asarray(polynomial_score_val).reshape((6,1))
RBF_score_val=np.asarray(RBF_score_val).reshape((6,1))
sigmoid_score_val=np.asarray(sigmoid_score_val).reshape((6,1))

linear_score_test=np.asarray(linear_score_test).reshape((6,1))
polynomial_score_test=np.asarray(polynomial_score_test).reshape((6,1))
RBF_score_test=np.asarray(RBF_score_test).reshape((6,1))
sigmoid_score_test=np.asarray(sigmoid_score_test).reshape((6,1))
str1="Score on Linear Kernel"
plt.figure()
plt.title(str1)
plt.plot(c_list,linear_score_train,'r',label="Train")
plt.plot(c_list,linear_score_val,'b',label="Cross-Validation")
plt.plot(c_list,linear_score_test,'g',label="Test")

plt.xlabel('C Value')
plt.ylabel("Score")
plt.legend(loc='best')  
  

str1="Score on Polynomial Kernel"
plt.figure()
plt.title(str1)
plt.plot(c_list,polynomial_score_train,'r',label="Train")
plt.plot(c_list,polynomial_score_val,'b',label="Cross-Validation")
plt.plot(c_list,polynomial_score_test,'g',label="Test")

plt.xlabel('C Value')
plt.ylabel("Score")
plt.legend(loc='best')
   
   

str2=str1="Score on RBF Kernel"
plt.figure()
plt.title(str2)
plt.plot(c_list,RBF_score_train,'r',label="Train")
plt.plot(c_list,RBF_score_val,'b',label="Cross-Validation")
plt.plot(c_list,RBF_score_test,'g',label="Test")

plt.xlabel('C Value')
plt.ylabel("Score")
plt.legend(loc='best')
    

str3="Score on Sigmoid Kernel"
plt.figure()
plt.title(str3)
plt.plot(c_list,sigmoid_score_train,'r',label="Train")
plt.plot(c_list,sigmoid_score_val,'b',label="Cross-Validation")
plt.plot(c_list,sigmoid_score_test,'g',label="Test")

plt.xlabel('C Value')
plt.ylabel("Score")
plt.legend(loc='best')
plt.show()      
      
                   
