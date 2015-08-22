__author__ = 'Ariel'

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import svm

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
clf = RandomForestClassifier(n_estimators=trees)
clf = Pipeline([('feature_selection', LinearSVC()),

              ('classification', RandomForestClassifier(n_estimators=trees))])
# clf = svm.SVC(kernel = "poly", C= 150, degree=3)

success_sum = 0

total_cms = [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]]

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
        percent = 1.0 * len(successes) / len(predictions)
        model_success_sum += percent

        cm = confusion_matrix(y_test, predictions)
        for row in range(6):
            for col in range(6):
                total_cms[row][col] += cm[row][col]

    model_success_mean = model_success_sum / model_reps
    success_sum += model_success_mean

success_mean = success_sum / div_reps

print (success_mean)

print (total_cms)