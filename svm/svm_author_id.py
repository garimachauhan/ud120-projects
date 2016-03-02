#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()






#########################################################
### your code goes here ###
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

t0 = time()
#clf = svm.SVC(kernel = 'linear')
clf = svm.SVC(kernel = 'rbf', C = 10000.0)
clf.fit(features_train, labels_train)
print "training time:", time()-t0

t1 = time()
pred = clf.predict(features_test)
#answer=pred[10], pred[26], pred[50]
#print "10th, 26th and 50th element predictions:", answer
count = 0
for x in pred:
    if x == 1:
        count+=1
print "No of emails predicted to be Chris's:", count
print "prediction time:", time()-t1


print accuracy_score(pred, labels_test)
#########################################################


