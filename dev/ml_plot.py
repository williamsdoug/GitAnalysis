#
# ml_plot.py Machine Learner Plotting Routines and other helpers
#
# Author:  Doug Williams - Copyright 2015
#
# Last updated 6/16/2015
#
# History:
# 6/16/15 - Add support for random number seed param for reproducable results
#
# from ml_plot import plot_validation_curve, plot_learning_curve
# from ml_plot import get_datasets, eval_clf, eval_predictions
# from ml_plot import PredictCV, PredictCV_TrainTest
# from ml_plot import PredictCV_TrainTestValidate
# from ml_plot import my_plot_learning_curve
# from ml_plot import plot_prediction_curve
# from ml_plot import getClassifierProbs
# from ml_plot import plotThresholdDistribuition, plotPredictionStats
# from ml_plot import plotCombinedResults

import matplotlib.pyplot as plt
import numpy as np
import random
import sys

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
# from BugFixWorkflow import find_legacy_cutoff


class PredictCV(object):
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

    seed: : int, default=None
        Optional seed for random number generator
    """

    def __init__(self, n, history=200, future=50, ignore=0.1,
                 n_iter=10, seed=None):
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
        self.seed = seed

        if type(ignore) is float and ignore >= 0.0 and ignore < 1.0:
            self.n = int(n*(1.0 - ignore))
        elif type(ignore) is int and ignore >= 0 and ignore < int(n):
            self.n = int(n) - ignore
        else:
            raise Exception('Invalid value for ignore')

        if history + future >= self.n:
            print 'Warning: history+future exceeds n', history, future, int(n)
            history = self.n - future - 1
            self.history = history
            print '         clipping history to:', history
            # raise Exception('history + future exceeds n')

    def __iter__(self):
        if isinstance(self.seed, int):
            random.seed(self.seed)

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


def PredictCV_TrainTest(X, Y, **kwargs):
    for train_idx, test_idx in PredictCV(len(Y), **kwargs):
        yield X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]


def PredictCV_TrainTestValidate(X, Y, train_size=50, test_size=50,
                                validate_size=50, shuffle=True,
                                **kwargs):
    for combined_idx, validate_idx in \
        PredictCV(len(Y), history=(train_size+test_size),
                  future=validate_size, **kwargs):
        if shuffle:
            np.random.shuffle(combined_idx)
        train_idx = combined_idx[0:train_size]
        test_idx = combined_idx[train_size:]
        yield [X[train_idx], Y[train_idx],
               X[test_idx], Y[test_idx],
               X[validate_idx], Y[validate_idx]]


def get_dataset(project, importance):
    (combined_commits,
     legacy_cutoff) = commit_postprocessing(project, importance=importance)

    # legacy_cutoff = find_legacy_cutoff(combined_commits)
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


def eval_predictions(clf, X, Y, history_sizes=[], future_sizes=[],
                     n_iter=10, seed=None):
    all_results = []
    for history in history_sizes:
        for future in future_sizes:
            results = []
            cv = PredictCV(len(Y), history=history, future=future,
                           n_iter=n_iter, seed=seed)
            title = '** Predictions for hist=' + str(history) \
                    + ' future='+str(future) + ' **'
            for train_idx, test_idx in cv:
                X_train = X[train_idx]
                Y_train = Y[train_idx]
                X_test = X[test_idx]
                Y_test = Y[test_idx]
                clf.fit(X_train, Y_train)
                value = eval_clf(clf, X_test, Y_test, verbose=False)
                value['history'] = history
                value['future'] = future
                results.append(value)
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
                                          400, 500, 1000],
                           n_iter=10, seed=None):
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
                                              future=future, n_iter=n_iter,
                                              seed=seed),
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
                          future_sizes=[50, 100, 200, 300, 500],
                          n_iter=10, seed=None):
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
                                              future=f, n_iter=n_iter,
                                              seed=seed),
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


#
# Functions to explore a predictors probability Functions
#

def getClassifierProbs(clf, X, Y, history=2000, future=500,
                       n_iter=10, seed=None):
    """Gets probabilities for a given classifier"""
    results = []
    for (X_train, Y_train,
         X_test, Y_test) in PredictCV_TrainTest(X, Y,  history=history,
                                                future=future,
                                                n_iter=n_iter, seed=seed):

        clf.fit(X_train, Y_train)
        y_predict = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        yy_prob = y_prob[:, 1] - y_prob[:, 0]
        y_log_prob = clf.predict_proba(X_test)

        results.append({'Y_test': Y_test, 'y_predict': y_predict,
                        'y_prob_raw': y_prob, 'y_prob_net': yy_prob,
                        'y_log_prob': y_log_prob})
        print '*',
        sys.stdout.flush()
    return results


def applyThreshold(results, thresh, verbose=True):
    """Apply threshold to a predicted probabilities and compute statistics"""
    total_pos = 0
    total_TP = 0
    total_FP = 0
    total_FN = 0
    for r in results:
        Y_test = r['Y_test']
        # y_predict = r['y_predict']
        y_prob = r['y_prob_net']
        # y_log_prob = r['y_log_prob']

        total_pos += sum(Y_test)
        total_TP += sum([1 for i in range(len(Y_test))
                         # if y_prob[i, 1] >= thresh
                         if y_prob[i] >= thresh
                         # and Y_test[i] and y_predict[i]])
                         and Y_test[i]])
        total_FP += sum([1 for i in range(len(Y_test))
                         # if y_prob[i, 1] >= thresh
                         if y_prob[i] >= thresh
                         # and not Y_test[i] and y_predict[i]])
                         and not Y_test[i]])
        total_FN += sum([1 for i in range(len(Y_test))
                         # if y_prob[i, 1] >= thresh
                         if y_prob[i] < thresh
                         # and Y_test[i] and not y_predict[i]])
                         and Y_test[i]])
    try:
        recall = float(total_TP)/float(total_pos)
        precision = float(total_TP)/float(total_TP + total_FP)

        F1 = 2.0 * precision * recall / (precision + recall)
        if verbose:
            print 'Results for threshold:', thresh
            print ('TP: {0:4d}    FP: {1:4d} \nFN: {2:4d} \n\nTotalPos:{3:4d}'
                   .format(total_TP, total_FP, total_FN, total_pos))
            print ('Precision: {:0.2f}  Recall: {:0.2f},  F1: {:0.2f}'
                   .format(precision, recall, F1))
            print
            print
        return {'thresh': thresh, 'precision': precision,
                'recall': recall, 'F1': F1,
                'TP': total_TP, 'FP': total_FP, 'FN': total_FN}
    except Exception:
        if verbose:
            print 'Results for threshold:', thresh
            print '   *** Skipped due to divide by zero ***'
            print
        return None


def plotThresholdDistribuition(results):
    """Shows histograms of predicted probability values"""
    plt.figure(figsize=(8, 3))
    plt.title('Commits by Prediction')
    plt.xlabel('y_prob')
    plt.ylabel('commits')
    plt.hist([y for r in results for y in r['y_prob_net']], bins=40)
    plt.show()

    plt.figure(figsize=(8, 3))
    plt.title('Commits by Prediction for True')
    plt.xlabel('y_prob')
    plt.ylabel('commits')
    plt.hist([y for r in results for i, y in enumerate(r['y_prob_net'])
              if r['Y_test'][i]], bins=40)
    plt.show()

    plt.figure(figsize=(8, 3))
    plt.title('Commits by Prediction for False')
    plt.xlabel('y_prob')
    plt.ylabel('commits')
    plt.hist([y for r in results for i, y in enumerate(r['y_prob_net'])
              if not r['Y_test'][i]], bins=40)
    plt.show()


def plotPredictionStats(results, verbose=False):
    """Plots Precision, Recall and F1 based in predictor probabilities"""

    predictionStats = [applyThreshold(results, thresh, verbose=False)
                       for thresh in np.arange(1.0, -0.125, -0.025)]

    x = [stat['thresh'] for stat in predictionStats if stat]

    plt.figure(figsize=(8, 3))
    plt.title('Classifier Results vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('TP/FP/FN')
    plt.xlim(xmin=-0.10)
    plt.grid(True)
    y_TP = [stat['TP'] for stat in predictionStats if stat]
    y_FP = [stat['FP'] for stat in predictionStats if stat]
    y_FN = [stat['FN'] for stat in predictionStats if stat]
    plt.plot(x, y_TP, 'g', label='TP')
    plt.plot(x, y_FP, 'r', label='FP')
    plt.plot(x, y_FN, 'b', label='FN')
    plt.legend(loc='upper right', shadow=False)
    plt.show()

    plt.figure(figsize=(8, 3))
    plt.title('Precision, Recall and F1 vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall/F1')
    plt.xlim(xmin=-0.10)
    plt.grid(True)
    y_prec = [stat['precision'] for stat in predictionStats if stat]
    plt.plot(x, y_prec, 'g', label='Precision')
    y_recall = [stat['recall'] for stat in predictionStats if stat]
    plt.plot(x, y_recall, 'b', label='Recall')
    y_f1 = [stat['F1'] for stat in predictionStats if stat]
    plt.plot(x, y_f1, 'r', label='F1')
    plt.legend(loc='upper right', shadow=False)
    plt.show()


def plotCombinedResults(all_results, verbose=False):
    """Plots TP and FP for all predictors based on prediction probabilities"""

    all_meta = [{'marker': 'g', 'TP_marker': 'g', 'FP_marker': 'g--'},
                {'marker': 'r', 'TP_marker': 'r', 'FP_marker': 'r--'},
                {'marker': 'b', 'TP_marker': 'b', 'FP_marker': 'b--'},
                {'marker': 'm', 'TP_marker': 'm', 'FP_marker': 'm--'},
                {'marker': 'y', 'TP_marker': 'y', 'FP_marker': 'y--'},
                {'marker': 'k', 'TP_marker': 'k', 'FP_marker': 'k--'},
                ]

    predictionStats = {}
    for k in all_results.keys():
        predictionStats[k] = [applyThreshold(all_results[k],
                                             thresh, verbose=False)
                              for thresh in np.arange(1.0, -0.125, -0.025)]

    for ii in range(0, len(all_results), len(all_meta)):
        plt.figure(figsize=(8, 8))
        plt.title('Classifier Results vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('TP/FP/FN')
        plt.xlim(xmin=-0.10)
        plt.grid(True)
        for i, k in enumerate(predictionStats.keys()[ii:ii+len(all_meta)]):
            meta = all_meta[i]
            x = [stat['thresh'] for stat in predictionStats[k] if stat]
            y_TP = [stat['TP'] for stat in predictionStats[k] if stat]
            y_FP = [stat['FP'] for stat in predictionStats[k] if stat]
            # y_FN = [stat['FN'] for stat in predictionStats[k] if stat]
            plt.plot(x, y_TP, meta['TP_marker'], label='{} - TP'.format(k))
            plt.plot(x, y_FP, meta['FP_marker'], label='{} - FP'.format(k))
            # plt.plot(x, y_FN, 'b', label='FN')
        plt.legend(loc='upper right', shadow=False)
        plt.show()

    for ii in range(0, len(all_results), len(all_meta)):
        plt.figure(figsize=(8, 8))
        plt.title('Classifier Precision vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.xlim(xmin=-0.10)
        plt.grid(True)
        for i, k in enumerate(predictionStats.keys()[ii:ii+len(all_meta)]):
            meta = all_meta[i]
            x = [stat['thresh'] for stat in predictionStats[k] if stat]
            y = [stat['precision'] for stat in predictionStats[k] if stat]
            # y_FN = [stat['FN'] for stat in predictionStats[k] if stat]
            plt.plot(x, y, meta['marker'], label=k)
        plt.legend(loc='upper left', shadow=False)
        plt.show()


def plotPrecisionRecallHeader(small=False):
    """Helper function for protPrecisionRecall()"""
    if small:
        plt.figure(figsize=(5, 5))
    else:
        plt.figure(figsize=(8, 8))
    plt.title('Precision vs Recall')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.xlim(xmin=0.1, xmax=0.8)
    plt.ylim(ymin=0.1, ymax=0.8)


def plotPrecisionRecallCurve(result, marker='b', label=None):
    """Plots Precision vs Recall Curve for individual estimator"""

    predictionStats = [applyThreshold(result, thresh, verbose=False)
                       for thresh in np.arange(1.0, -0.525, -0.025)]

    y_prec = [stat['precision'] for stat in predictionStats if stat]
    y_recall = [stat['recall'] for stat in predictionStats if stat]
    plt.plot(y_prec, y_recall, marker, label=label)


def plotPrecisionRecall(result):
    """Top level - plots precision vs recall for an individual result"""

    plotPrecisionRecallHeader(small=True)
    plotPrecisionRecallCurve(result)
    # plt.legend(loc='upper right', shadow=False)
    plt.show()


def plotAllPrecisionRecall(results, keys=[]):
    """Top level - plots precision vs recall for a set of results"""
    if not keys:
        keys = results.keys()
    markers = ['r', 'b', 'g', 'y', 'c', 'm', 'k',
               'r:', 'b:', 'g:', 'y:', 'c:', 'm:', 'k:',
               'r--', 'b--', 'g--', 'y--', 'c--', 'm--', 'k--', ]

    plotPrecisionRecallHeader()
    for i, (k, v) in enumerate([(k, v) for k, v in results.items()
                                if k in keys]):
            plotPrecisionRecallCurve(v, marker=markers[i], label=k)
    plt.legend(loc='upper right', shadow=False)
    plt.show()


def showScatterProb(firstKey, secondKey, all_results):
    """Scatter plot of two sets of probability predictions,
    color signifies growntruth """
    groundTruth = np.concatenate([r['Y_test']
                                  for r in all_results[firstKey]])
    firstProbs = np.concatenate([r['y_prob_net']
                                 for r in all_results[firstKey]])
    secondProbs = np.concatenate([r['y_prob_net']
                                  for r in all_results[secondKey]])

    fig = plt.figure(figsize=(8, 8))
    plt.title('Ground Truth g = 1, r = 0')
    plt.scatter([firstProbs[i]
                 for i, gt in enumerate(groundTruth) if not gt],
                [secondProbs[i]
                 for i, gt in enumerate(groundTruth) if not gt],
                marker='o', c='r', alpha=0.01)
    plt.scatter([firstProbs[i]
                 for i, gt in enumerate(groundTruth) if gt],
                [secondProbs[i]
                 for i, gt in enumerate(groundTruth) if gt],
                marker='o', c='g', alpha=0.01)
    plt.xlabel(firstKey)
    plt.ylabel(secondKey)
    plt.plot([0, 0], [-1, 1], 'r')
    plt.plot([-1, 1], [0, 0], 'r')
    plt.show()
