from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_roc_curve(y_true, y_score):
    """Plot ROC curve.

    Parameters
    ----------
    y_true: ground true labels
    y_score: predicted score
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(6.5, 3.5), dpi=100)
    x = fpr.tolist()
    y = tpr.tolist()
    plt.scatter(
        x, y, color='#1f497d', s=5, marker='+',
        label='AUC = ' + str(round(roc_auc, 4))
    )
    plt.ylabel('True Positive Rate', color='#1f497d')
    plt.xlabel('False Positive Rate', color='#1f497d')
    plt.title('ROC curve', color='#1f497d')
    plt.grid(color='gray', linestyle='dashed')
    plt.plot([0, 1], [0, 1], 'r--', color='black', label='random model')
    plt.plot([0, 1], [1, 1], 'r--', color='#057DA1', label='optimal model')
    plt.xlim(-0.1, 1.1)
    plt.ylim(0, 1.1)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.legend(loc='lower right', fontsize=10)

    plt.show()


def plot_precision_recall_curve(y_true, y_score):
    """Plot Precision-recall curve.

    Parameters
    ----------
    y_true: ground true labels
    y_score: predicted score
    """
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    plt.figure(figsize=(6.5, 3.5), dpi=100)
    x = rec.tolist()
    y = prec.tolist()
    plt.scatter(
        x, y, color='#1f497d', s=5, marker='+',
        label='AP = ' + str(round(average_precision, 4))
    )
    plt.ylabel('Precision', color='#1f497d')
    plt.xlabel('Recall', color='#1f497d')
    plt.title('Precision-Recall curve', color='#1f497d')
    plt.grid(color='gray', linestyle='dashed')
    plt.xlim(-0.1, 1.1)
    plt.ylim(0, 1.1)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.legend(loc='lower right', fontsize=10)

    plt.show()


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in.

    Used in plot_with_decision_boundaries function.

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1] >= 0.5
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_with_decision_boundaries(x0, x1, y, clf, title=None, ax=None, x0_label="", x1_label="", figsize=(7,6)):
    xx, yy = make_meshgrid(x0, x1)
    y_cat = pd.Series(y).astype("category").cat.codes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(x0, x1, c=y_cat, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)
    # ax.set_xticks(())
    # ax.set_yticks(())
    if title is None:
        title = type(clf)
    ax.set_title(title)
