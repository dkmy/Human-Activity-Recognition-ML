__author__ = 'DavidYang'

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

path='/Users/DavidYang/Desktop/Machine Learning/UCI HAR Dataset'
pathTrain='/Users/DavidYang/Desktop/Machine Learning/UCI HAR Dataset/train/'

X_trainUni = np.loadtxt(pathTrain+"X_train.csv",delimiter=',')
y_trainUni = np.loadtxt(pathTrain+"y_train.txt",delimiter="\n")

clfUni = svm.SVC(kernel = 'poly', C=150, degree=3)
clfUni.fit(X_trainUni, y_trainUni)
# clfUni = Pipeline([('feature_selection', LinearSVC()),
#                ('classification', RandomForestClassifier(n_estimators=200))])
# clfUni.fit(X_trainUni, y_trainUni)



clfPersonal = svm.SVC(kernel = 'rbf', C=150, degree=2)

# clfPersonal = Pipeline([('feature_selection', LinearSVC()),
#                ('classification', RandomForestClassifier(n_estimators=200))])

falsePositives = 0
totalCount =0
def TrainPersonal(n):
    tempPath = path+"/together/SortedSubjects/subject " + str(n) + ".csv"
    subject1 = np.loadtxt(tempPath, delimiter=',')
    subject_y = subject1[:, 1]
    subject_x = subject1[:, 2:]
    # X_train, X_test, y_train, y_test = train_test_split(subject_x, subject_y, test_size = 0.2, random_state = 1341)
    clfPersonal.fit(subject_x,subject_y)

def testedBy(n):
    tempPath = path+"/together/SortedSubjects/subject " + str(n) + ".csv"
    subject1 = np.loadtxt(tempPath, delimiter = ',')
    subject_x = subject1[:, 2:]
    errors = 0
    cases=0
    for i in range(len(subject_x)):
        a  = clfUni.predict(subject_x[i])
        if a == 1 or a==2 or a==3:
            cases = cases+1
            if clfUni.predict(subject_x[i]) != clfPersonal.predict(subject_x[i]):
                errors = errors + 1
    print ("self test by subject ", n, "errors: " ,errors, "cases: " ,cases, "accuracy: ", errors/cases)


# for i in [1,3,5,6,7,8,11,14,15,16,17,19,21,22,23,25,26,27,28,29,30]:
for i in [2,4,9,10,12,13,18,20,24]:

    TrainPersonal(i)
    testedBy(i)
