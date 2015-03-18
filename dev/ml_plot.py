# Plotting Routines and other helpers
#
# from ml_plot import plot_validation_curve, plot_learning_curve
# from ml_plot import get_datasets, eval_clf, eval_predictions
# from ml_plot import PredictCV
# from ml_plot import my_plot_learning_curve
# from ml_plot import plot_prediction_curve

import matplotlib.pyplot as plt
import numpy as np
import math
import random

from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

from Git_Extract import get_commit_ordering_min_max
from commit_analysis import fit_features
from commit_analysis import extract_features
from commit_analysis import autoset_threshold
from BugFixWorkflow import compute_selected_bug_fixes
from BugFixWorkflow import commit_postprocessing
from BugFixWorkflow import find_legacy_cutoff


class PredictCV():
    """Prediction cross validation iterator.

    Training based on prior history, testing based on immediately
    subsequent values.

    For use with a feature matrix that has been sorted in ascending order
    based on time, randomly picks a reference point.  Training set is
    composed of features immediately prior to reference point, whereas
    Test set is composed of features immediately following reference point.

    Parameters
    ----------
    n : int
        Total number of elements.

    history : int, default=200
        Training set size (sequential features)

    future : int, default=50
        Test set size.  Sequential features immediately following
        Training set.

    ignore : int of float
        Specifies either number of fraction of most recent features to be
        excluded from training and test sets.  Defaults to 10%

    n_iter : int, default=10
        Number of folds. Must be at least 2.
    """

    def __init__(self, n, history=200, future=50, ignore=0.1, n_iter=10):
        if n <= 0:
            raise Exception('Invalid value for n')
        if n_iter <= 0:
            raise Exception('Invalid value for n_iter')
        if history <= 1:
            raise Exception('history must be positive integer')
        if future <= 1:
            raise Exception('future must be positive integer')

        self.n = int(n)
        self.history = history
        self.future = future
        self.n_iter = n_iter

        if type(ignore) is float and ignore >= 0.0 and ignore < 1.0:
            self.n = int(n*(1.0 - ignore))
        elif type(ignore) is int and ignore >= 0 and ignore < int(n):
            self.n = int(n) - ignore
        else:
            raise Exception('Invalid value for ignore')

        if history + future >= self.n:
            raise Exception('history + future exceeds n')

    def __iter__(self):
        for i in range(self.n_iter):
            start = random.randint(self.history, self.n-self.future)
            yield np.arange(start - self.history, start), \
                np.arange(start, start + self.future)

    def __repr__(self):
        return '%s.%s(n=%i, n_iter=%i, history=%i, future=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
            self.n_iter,
            self.history,
            self.future,
        )

    def __len__(self):
        return self.n_iter


def get_dataset(project, importance):
    combined_commits = commit_postprocessing(project, importance=importance)

    legacy_cutoff = find_legacy_cutoff(combined_commits)
    min_order, max_order = get_commit_ordering_min_max(combined_commits)
    actual_bugs = compute_selected_bug_fixes(combined_commits,
                                             legacy_cutoff=legacy_cutoff,
                                             min_order=min_order,
                                             max_order=max_order)
    guilt_threshold, labeled_bugs = autoset_threshold(combined_commits,
                                                      actual_bugs)
    print 'Setting guilt threshold to:', guilt_threshold
    print 'Labeled bugs:', labeled_bugs, ' vs Actual bugs:', actual_bugs

    extract_state = fit_features(combined_commits, min_order=min_order,
                                 max_order=max_order)

    _, Y, X, col_names = \
        extract_features(combined_commits, extract_state,
                         min_order=min_order,
                         max_order=max_order, threshold=guilt_threshold)
    return Y, X


def eval_clf(clf, X, Y, verbose=True, title=False):
    Y_predict = clf.predict(X)

    f1 = metrics.f1_score(Y, Y_predict)
    accuracy = metrics.accuracy_score(Y, Y_predict)
    precision = metrics.precision_score(Y, Y_predict)
    recall = metrics.recall_score(Y, Y_predict)
    confusion = metrics.confusion_matrix(Y, Y_predict)

    if verbose:
        if title:
            print title
            print
        print 'F1:', f1
        print 'accuracy:', accuracy
        print 'precision:', precision
        print 'recall:', recall
        print 'confusion matrix'
        print confusion
        print
        print metrics.classification_report(Y, Y_predict)

    return {'f1': f1, 'accuracy': accuracy,
            'precision': precision, 'recall': recall,
            'confusion': confusion}


def eval_predictions(clf, X, Y, history_sizes=[], future_sizes=[]):
    all_results = []
    for history in history_sizes:
        for future in future_sizes:
            results = []
            cv = PredictCV(len(Y), history=history, future=future, n_iter=10)
            title = '** Predictions for hist=' + str(history) \
                    + ' future='+str(future) + ' **'
            for train_idx, test_idx in cv:
                X_train = X[train_idx]
                Y_train = Y[train_idx]
                X_test = X[test_idx]
                Y_test = Y[test_idx]
                clf.fit(X_train, Y_train)
                val = eval_clf(clf, X_test, Y_test, verbose=False)
                val['history'] = history
                val['future'] = future
                results.append(val)
            # Show aggregated results
            print title
            print
            print 'F1:        {0:0.2f}  +/- {1:0.2f}'.format(
                np.mean([val['f1'] for val in results]),
                np.std([val['f1'] for val in results]))
            print 'Accuracy:  {0:0.2f}  +/- {1:0.2f}'.format(
                np.mean([val['accuracy'] for val in results]),
                np.std([val['accuracy'] for val in results]))
            print 'Precision: {0:0.2f}  +/- {1:0.2f}'.format(
                np.mean([val['precision'] for val in results]),
                np.std([val['precision'] for val in results]))
            print 'Recall:    {0:0.2f}  +/- {1:0.2f}'.format(
                np.mean([val['recall'] for val in results]),
                np.std([val['recall'] for val in results]))
            print
            print 'Aggregate Confusion Matrix -', len(results), 'iterations'
            print np.array([[sum([val['confusion'][0, 0]
                                  for val in results]),
                             sum([val['confusion'][0, 1]
                                  for val in results])],
                            [sum([val['confusion'][1, 0]
                                  for val in results]),
                             sum([val['confusion'][1, 1]
                                  for val in results])]])
            print
            print
            all_results.append(results)
    return all_results

#
# Plotting routines below derived from examples at: http://scikit-learn.org/
#


def plot_validation_curve(estimator, X, y, param_name, param_range,
                          title='', ylim=None, cv=3, scoring="accuracy",
                          n_jobs=1, scale='log'):

    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    if scale == 'log':
        plt.semilogx(param_range, train_scores_mean, label="Training score",
                     color="r")
    else:
        plt.plot(param_range, train_scores_mean, label="Training score",
                 color="r")

    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="r")
    if scale == 'log':
        plt.semilogx(param_range, test_scores_mean,
                     label="Cross-validation score",
                     color="g")
    else:
        plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")

    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()

    # find best result
    best_score = 0
    best_idx = 0
    for i, score in enumerate(test_scores_mean):
        if score > best_score:
            best_score = score
            best_idx = i
    return best_score, param_range[best_idx], param_name


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
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


def my_plot_learning_curve(estimator, title, X, y, ylim=None,
                           n_jobs=1, future=100, scoring='f1',
                           history_sizes=[50, 100, 200, 300,
                                          400, 500, 1000]):
    """
    Generate a simple plot of the test learning curve.

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

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    history_sizes : list, optional
        Sizes of training set

    future : int, optional
        Size of test set used to evaluate predictive capability

    scoring : string, optional
        Scoring metric
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    #
    # Replace learning_curve wth
    #
    test_scores_mean = []
    test_scores_std = []
    for h in history_sizes:
        scores = cross_val_score(estimator, X, y=y, scoring=scoring,
                                 cv=PredictCV(len(y), history=h,
                                              future=future, n_iter=10),
                                 n_jobs=n_jobs)

        test_scores_mean.append(np.mean(scores))
        test_scores_std.append(np.std(scores))
    test_scores_mean = np.array(test_scores_mean)
    test_scores_std = np.array(test_scores_std)
    plt.grid()

    plt.fill_between(history_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(history_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_prediction_curve(estimator, title, X, y, ylim=None,
                          n_jobs=1, history=500, scoring='f1',
                          future_sizes=[50, 100, 200, 300, 500]):
    """
    Generate a simple plot of the test prediction curve.

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

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    history : int, optional
        Size of training set

    future_sizes : list
        Size of test set

    scoring : string, optional
        Scoring metric
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Future window size")
    plt.ylabel("Score")

    #
    # Replace learning_curve wth
    #
    test_scores_mean = []
    test_scores_std = []
    for f in future_sizes:
        scores = cross_val_score(estimator, X, y=y, scoring=scoring,
                                 cv=PredictCV(len(y), history=history,
                                              future=f, n_iter=10),
                                 n_jobs=n_jobs)

        test_scores_mean.append(np.mean(scores))
        test_scores_std.append(np.std(scores))
    test_scores_mean = np.array(test_scores_mean)
    test_scores_std = np.array(test_scores_std)
    plt.grid()

    plt.fill_between(future_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(future_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
