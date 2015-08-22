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

pathTrain='/Users/DavidYang/Desktop/Machine Learning/UCI HAR Dataset/train/'
pathTest = '/Users/DavidYang/Desktop/Machine Learning/UCI HAR Dataset/test/'
X_train = np.loadtxt(pathTrain+"X_train.csv",delimiter=',')
y_train = np.loadtxt(pathTrain+"y_train.txt",delimiter="\n")
X_test = np.loadtxt(pathTest + "X_test.csv",delimiter = ',')
y_test = np.loadtxt(pathTest + "y_test.csv",delimiter = "\n")

#the following SVC can be changed for different values
kf_total = cross_validation.KFold(len(X_train), n_folds=10,  shuffle=True, random_state=32)

accuracy = 0;
clf = svm.SVC(kernel = "poly", C= 150, degree=3)

cross_validation.cross_val_score(clf, X_train, y_train, cv=kf_total, n_jobs = 1)
scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=kf_total, n_jobs = 1)

# for train, test in kf_total:
# # clf = svm.LinearSVC(penalty = 'l2', loss = 'l2', dual = False, C=1000)
# #     clf = svm.SVC(kernel = 'poly', C=400, degree=2)
# #     clf = svm.LinearSVC()
#     clf.fit(X_train[train], y_train[train])
#     errors = 0
#     for i in range(len(test)):
#         if clf.predict(X_train[test[i]]) != y_train[test[i]]:
#             errors = errors + 1
#     accuracy = accuracy + (1-errors/len(test))*100
#     # cm = confusion_matrix(y_test, y_pred)


print (sum(scores)/10)
# print ("The mean accuracy is " ,accuracy/5)
