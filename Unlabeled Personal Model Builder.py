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


X_trainUni = np.loadtxt(pathTrain+"X_train.csv",delimiter=',')
y_trainUni = np.loadtxt(pathTrain+"y_train.txt",delimiter="\n")
clfUni = svm.SVC(kernel = 'rbf', C=60, degree=3)
clfUni.fit(X_trainUni, y_trainUni)
overallAccuracy = 0

def Builder(n):
    tempPath = path+"/together/SortedSubjects/subject " + str(n) + ".csv"
    subject1 = np.loadtxt(tempPath, delimiter=',')
    subject_y = subject1[:, 1]
    subject_x = subject1[:, 2:]
    X_train, X_test, y_train, y_test = train_test_split(subject_x, subject_y, test_size = 0.2, random_state = 1341)

    temp = []
    for i in range(len(X_train)):
        temp.append(clfUni.predict(X_train[i]))
        # y_trianBuilt  = np.append(y_trainBuilt, clfUni.predict(X_train[i]))
    y_trainBuilt = np.array(temp)
    np.ravel(y_trainBuilt)


    clfPersonal = svm.SVC(kernel='rbf', C=60, degree=3)
    clfPersonal.fit(X_train, y_trainBuilt)
    errors = 0
    for i in range(len(X_test)):
        if clfPersonal.predict(X_test[i]) != y_test[i]:
            errors = errors + 1

    print ("Subject " , str(n), "accuracy: " ,100*(1-errors/len(X_test)))
    return 100*(1-errors/len(X_test))

overallAccuracy = overallAccuracy+Builder(4)
overallAccuracy = overallAccuracy+Builder(2)
overallAccuracy = overallAccuracy+ Builder(9)
overallAccuracy = overallAccuracy+ Builder(10)
overallAccuracy = overallAccuracy+ Builder(12)
overallAccuracy = overallAccuracy+ Builder(13)
overallAccuracy = overallAccuracy+ Builder(18)
overallAccuracy = overallAccuracy+ Builder(20)
overallAccuracy = overallAccuracy+ Builder(24)

print (overallAccuracy/9)





