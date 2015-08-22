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
    X_train, X_test, y_train, y_test = train_test_split(subject_x, subject_y, test_size = 0.2, random_state = 1341)

    # clf = svm.SVC(kernel='rbf', C=300, degree=3)
    # clf.fit(X_train, y_train)
    # errors = 0
    # for i in range(len(X_test)):
    #     if clf.predict(X_test[i]) != y_test[i]:
    #         errors = errors + 1
    # accuracy = 0
    # return (100*(1-errors/len(X_test)))

#the following SVC can be changed for different values
    kf_total = cross_validation.KFold(len(X_train), n_folds=10,  shuffle=True, random_state=43)
    # clf = svm.LinearSVC(penalty='l2', loss='l2', dual=False)
    accuracy = 0;
    clf = svm.SVC(kernel = 'rbf', C=150, degree=2)

    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=kf_total, n_jobs = 1)

    # for train, test in kf_total:
    #     # clf = svm.LinearSVC(penalty = 'l2', loss = 'l2', dual = False, C=1000)
    #     clf.fit(X_train[train], y_train[train])
    #     errors = 0
    #     for i in range(len(test)):
    #         if clf.predict(X_train[test[i]]) != y_train[test[i]]:
    #             errors = errors + 1
    #     accuracy = accuracy + (1-errors/len(test))*100
    # # cm = confusion_matrix(y_test, y_pred)
    # print (scores)

    return sum(scores)/10

counter = 0
#personal
for i in range(1,31):
    print ("Subject ",str(i), "accuracy is :" , Test(i))
    counter = counter + Test(i)

print ("The mean accuracy is " ,counter/30)
