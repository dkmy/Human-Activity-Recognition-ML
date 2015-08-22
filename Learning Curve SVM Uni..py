__author__ = 'DavidYang'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve



path='/Users/DavidYang/Desktop/Machine Learning/UCI HAR Dataset'
pathTrain='/Users/DavidYang/Desktop/Machine Learning/UCI HAR Dataset/train/'

# X_trainUni = np.loadtxt(pathTrain+"X_train.csv",delimiter=',')
# y_trainUni = np.loadtxt(pathTrain+"y_train.txt",delimiter="\n")
clfUni = svm.SVC(kernel = 'poly', C=150, degree=3)
# clfUni = Pipeline([('feature_selection', LinearSVC()),
#                ('classification', RandomForestClassifier(n_estimators=200))])
# clfUni.fit(X_trainUni, y_trainUni)
n=4

tempPath = path+"/together/together_smart.csv"
subject1 = np.loadtxt(tempPath, delimiter=',')
subject_y = subject1[:, 1]
subject_x = subject1[:, 2:]
X_train, X_test, y_train, y_test = train_test_split(subject_x, subject_y, test_size = 0.2, random_state = 123)

X = X_test
y = y_test

# X_train, X_test, y_train, y_test = train_test_split(subject_x, subject_y, test_size = 0.2, random_state = 123)
# clf.fit(X_train, y_train)

# estimator = Pipeline([('feature_selection', LinearSVC()),
#                ('classification', RandomForestClassifier(n_estimators=200))])
estimator = clfUni


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 15)):

    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

title = "Universal Model Learning Curve"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.

# tempPath = path+"/together/SortedSubjects/subject " + str(4) + ".csv"
# subject1 = np.loadtxt(tempPath, delimiter=',')
#
# subject_y = subject1[:, 1]
# subject_x = subject1[:, 2:]
# cv = cross_validation.ShuffleSplit(subject_x.shape[0], n_iter=100, test_size=0.2, random_state=0)
cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=80, test_size=0.2, random_state=0)



plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=6)

plt.show()