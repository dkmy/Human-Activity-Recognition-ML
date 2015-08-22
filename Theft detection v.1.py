__author__ = 'DavidYang'

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix



path='/Users/DavidYang/Desktop/Machine Learning/UCI HAR Dataset/'

given_X_train = np.loadtxt(path+'train/X_train.txt')
given_y_train = np.loadtxt(path+'train/y_train.txt')
given_subj_train = np.loadtxt(path+'train/subject_train.txt')

given_X_test = np.loadtxt(path+'test/X_test.txt')
given_y_test = np.loadtxt(path+'test/y_test.txt')
given_subj_test = np.loadtxt(path+'test/subject_test.txt')

X_smart = np.append(given_X_train, given_X_test, axis=0)
y_smart = np.append(given_y_train, given_y_test, axis=0)
subj_smart = np.append(given_subj_train, given_subj_test, axis=0)

'''
Train (50%)
Validation (30%)
Test (20%)
'''

div_reps = 5
model_reps = 10
trees = 200
# clf = RandomForestClassifier(n_estimators=trees)
clf = Pipeline([('feature_selection', LinearSVC()),
               ('classification', RandomForestClassifier(n_estimators=trees))])

success_sum = 0

for div_rep in range(div_reps):
    X_train, X_t, y_train, y_t, subj_train, subj_t = train_test_split(X_smart, y_smart, subj_smart, train_size=0.5)
    X_valid, X_test, y_valid, y_test, subj_train, subj_test = train_test_split(X_t, y_t, subj_t, test_size=0.4)
    model_success_sum = 0
    for model_rep in range(model_reps):
        X_train_valid = np.append(X_train, X_valid, axis=0)
        y_train_valid = np.append(y_train, y_valid, axis=0)
        clf.fit(X_train_valid, y_train_valid)
        predictions = clf.predict(X_test)
        successes = np.where((predictions == y_test) == True)[0]

    model_success_mean = model_success_sum / model_reps
    success_sum += model_success_mean

success_mean = success_sum / div_reps

print (success_mean)


path='/Users/DavidYang/Desktop/Machine Learning/UCI HAR Dataset'
pathTrain='/Users/DavidYang/Desktop/Machine Learning/UCI HAR Dataset/train/'

X_trainUni = np.loadtxt(pathTrain+"X_train.csv",delimiter=',')
y_trainUni = np.loadtxt(pathTrain+"y_train.txt",delimiter="\n")
clfUni = svm.SVC(kernel = 'rbf', C=60, degree=3)
clfUni.fit(X_trainUni, y_trainUni)

clfPersonal = svm.SVC(kernel='rbf', C=60, degree=3)

def TrainPersonal(n):
    tempPath = path+"/together/SortedSubjects/subject " + str(n) + ".csv"
    subject1 = np.loadtxt(tempPath, delimiter=',')
    subject_y = subject1[:, 1]
    subject_x = subject1[:, 2:]
    X_train, X_test, y_train, y_test = train_test_split(subject_x, subject_y, test_size = 0.2, random_state = 1341)

    clfPersonal.fit(X_train, y_train)
    return clfPersonal



def TheftBy(n,k):
    tempPath = path+"/together/SortedSubjects/subject " + str(n) + ".csv"
    subject1 = np.loadtxt(tempPath, delimiter = ',')
    subject_y = subject1[:, 1]
    subject_x = subject1[:, 2:]
    errors = 0
    cases = 0
    uniPredicts=[]
    for i in range(len(subject_x)):
        uniPredicts.append(clfUni.predict(subject_x[i]))
    for i in range(len(subject_x)):
        if uniPredicts[i] != clfPersonal.predict(subject_x[i]):
            errors = errors + 1
        cases = cases + 1

    if errors/cases > .15:
        # print (str(n) , ", theft detected in ", cases, ". error rate " ,errors/cases)
        if n==k:
            print ("False positive from user ", str(k))
        return 0
    # print (str(n), " correct user")
    return 1

average =0
for i in range(1,31):
    TrainPersonal(i)
    errors = 0
    falseNegative=0
    for j in range (1,31):
        errors = errors + TheftBy(j,i)

    print ("Subject : ",str(i), " False negatives: " ,errors-1)
    average = average + errors-1

print ("Average false positives: ", average)

