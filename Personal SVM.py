__author__ = 'DavidYang'

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets, linear_model, cross_validation, grid_search
from sklearn.metrics import confusion_matrix

path='/Users/DavidYang/Desktop/Machine Learning/UCI HAR Dataset'


def Test(n):
    tempPath = path+"/together/SortedSubjects/subject " + str(n) + ".csv"
    subject1 = np.loadtxt(tempPath, delimiter=',')
    subject_y = subject1[:, 1]
    subject_x = subject1[:, 2:]
    X_train, X_test, y_train, y_test = train_test_split(subject_x, subject_y, test_size = 0.2, random_state = 123)
    clf = svm.SVC(kernel = 'rbf', C=150, degree=2)
    clf.fit(X_train, y_train)

    errors = 0;

    for i in range(len(X_test)):
        if y_test[i] != clf.predict(X_test[i]):
            errors = errors + 1
    cm = confusion_matrix(y_test, clf.predict(X_test))

    for row in range(6):
            for col in range(6):
                sum_cms[row][col] = sum_cms[row][col]+ cm[row][col]
    print ("Subject ", str(n), " accuracy: ", ((1-errors/len(X_test))*100))
    return ((1-errors/len(X_test))*100)
counter = 0
sum_cms = [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]]

for i in range(1,31):
    counter = counter + Test(i)

print ("The mean accuracy is " ,counter/30)
print (sum_cms)
