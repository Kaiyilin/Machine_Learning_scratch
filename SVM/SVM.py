from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()
X=iris.data
y=iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = svm.SVC(kernel='rbf', C=0.5, gamma='auto')
clf2 = svm.SVC(kernel='sigmoid', C=0.5, gamma='auto')

clf.fit(X_train,y_train)
clf2.fit(X_train,y_train)


clf.predict(X_test)
clf2.predict(X_test)


print('kernel: rbf result')
print(clf.score(X_train,y_train))
print(clf.score(X_test, y_test))

print("----"*20)

print('kernel: sigmoid result')
print(clf.score(X_train,y_train))
print(clf.score(X_test, y_test))

