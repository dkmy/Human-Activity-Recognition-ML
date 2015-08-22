__author__ = 'Ariel'

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
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
# clf = svm.SVC(kernel = 'rbf', C=150, degree=2)



total_success_sum = 0  # sum of success rates for all subjects

sum_cms = []  # list of sum confusion matrices

for subj in range(30):
    where = np.where(subj_smart == subj+1)[0]
    subj_start = where[0]
    subj_end = where[-1] + 1

    subj_cms = []  # list of confusion matrices for subject

    div_success_sum = 0  # sum of success rates for all reps
    for div_rep in range(div_reps):
        # initialize lists of indices
        train = []
        valid = []
        test = []

        for act in range(6):
            act_where = np.where(y_smart[subj_start:subj_end] == act+1)[0]
            act_where = [i+subj_start for i in act_where]
            i_train, i_t = train_test_split(act_where, train_size=0.5)
            i_valid, i_test = train_test_split(i_t, test_size=0.4)
            train = np.append(train, i_train)
            valid = np.append(valid, i_valid)
            test = np.append(test, i_test)

        X_train = np.vstack([X_smart[i] for i in train])
        X_valid = np.vstack([X_smart[i] for i in valid])
        X_test = np.vstack([X_smart[i] for i in test])

        y_train = np.ravel(np.vstack([y_smart[i] for i in train]))
        y_valid = np.ravel(np.vstack([y_smart[i] for i in valid]))
        y_test = np.ravel(np.vstack([y_smart[i] for i in test]))

        subj_train = np.vstack([subj_smart[i] for i in train])
        subj_valid = np.vstack([subj_smart[i] for i in valid])
        subj_test = np.vstack([subj_smart[i] for i in test])

        ##### Statistics #####
        # print subj+1, ": sum-", len(where), ", train-", len(train), ", valid-", len(valid), ", test-", len(test)

        # ### Success on Valid Dataset #####
        # model_success_sum = 0
        # for model_rep in range(model_reps):
        #     clf.fit(X_train, y_train)
        #     predictions = clf.predict(X_valid)
        #     successes = np.where((predictions == y_valid) == True)[0]
        #     percent = 1.0 * len(successes) / len(predictions)
        #     model_success_sum += percent
        #
        #     cm = confusion_matrix(y_valid, predictions)
        #     subj_cms += [cm]

        #### Success on Test Dataset #####
        X_train_valid = np.append(X_train, X_valid, axis=0)
        y_train_valid = np.append(y_train, y_valid, axis=0)
        model_success_sum = 0
        for model_rep in range(model_reps):
            clf.fit(X_train_valid, y_train_valid)
            predictions = clf.predict(X_test)
            successes = np.where((predictions == y_test) == True)[0]
            percent = 1.0 * len(successes) / len(predictions)
            model_success_sum += percent

            cm = confusion_matrix(y_test, predictions)
            subj_cms += [cm]

        mean_model_success = 1.0 * model_success_sum / model_reps

        div_success_sum += mean_model_success

    mean_div_success = 1.0 * div_success_sum / div_reps
    print (subj+1, ": ", mean_div_success)

    total_success_sum += mean_div_success

    sum_subj_cms = [[0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]]
    for cm in subj_cms:
        for row in range(6):
            for col in range(6):
                sum_subj_cms[row][col] += cm[row][col]  # 1.0 * cm[row][col] / (div_reps * model_reps)
                # sum_subj_cms[row][col] = round(sum_subj_cms[row][col], 2)
    print (sum_subj_cms)

    sum_cms += [sum_subj_cms]

print ("mean: ", total_success_sum / 30.0)

total_cms = [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]]
for sum_cm in sum_cms:
    for row in range(6):
        for col in range(6):
            total_cms[row][col] += sum_cm[row][col]
            # total_cms[row][col] = round(total_cms[row][col], 2)
print (total_cms)
np.savetxt("confusion matrix for personal svm test dataset.csv" ,total_cms, delimiter = ",")
