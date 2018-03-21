"""
Training model for detect DGA domains using machine learning
"""
# from collections import OrderedDict
# from itertools import product
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
from sklearn.externals import joblib
# from matplotlib import pyplot as plt
# from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np

from config import training_data


def train():
    np.seterr(divide='ignore', invalid='ignore')
    # Overall training data.
    all_data_dict = pd.concat([training_data['legit'], training_data['dga']], ignore_index=True)

    # Calculate length for each domain.
    # Domains of any type (legit, dga) with length <6 receive very close results in the rating.
    # Also usually DGA domains have length > 6.
    # Based on this, they will not be taken into account in the model.
    all_data_dict['length'] = [len(x) for x in all_data_dict['domain']]
    all_data_dict = all_data_dict[all_data_dict['length'] > 6]

    # Calculate entropy for each domain.
    # all_data_dict['entropy'] = [entropy(x) for x in all_data_dict['domain']]

    # Create model to definite matrix of ngram counts.
    # Min frequency = 0.001% for domains in overall data.
    # Remark: for 0.01% appears a lot zero values on occurrence ngram.
    vectorizer = CountVectorizer(ngram_range=(3, 5), analyzer='char', max_df=1.0, min_df=0.0001)

    # Tokenize and count the ngram occurrences of legit domains.
    # Result of sparse matrix (most of the entries are zero).
    # "(x,y) n" mean that "(row, column) value".
    # Value - the number of times a ngram appeared in the domains
    # represented by the row of the matrix.
    ngram_matrix = vectorizer.fit_transform(training_data['legit']['domain'])

    # Transform to dense matrix (column sum),
    # then to multidimensional homogeneous array.
    ngram_counts = ngram_matrix.sum(axis=0).getA1()

    # Relation extracted ngram out of raw domains in out sparse transpose matrix
    # with ngram_counts through multiplication of vectors.
    all_data_dict['occur_ngrams'] = ngram_counts * vectorizer.transform(all_data_dict['domain']).transpose()

    # Array x holdings the training samples.
    # Array y holdings the target values (type labels) for the training samples.
    X = all_data_dict.as_matrix(['length', 'occur_ngrams'])
    y = np.array(all_data_dict['type'].tolist())

    # For the test, 20% of the original data is allocated.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=0)

    # Fit a model.
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X_train, y_train)

    # scores = cross_val_score(clf, X_test, y_test)
    # print(scores.mean())

    # Save model
    joblib.dump(clf, 'input data/model.pkl')

    # # Test for the effective selection of the number of trees
    # ensemble_clfs = [("RandomForestClassifier",
    #                   RandomForestClassifier(oob_score=True, random_state=1))]
    #
    # error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
    #
    # min_estimators = 20
    # max_estimators = 100
    #
    # for label, clf in ensemble_clfs:
    #     for i in range(min_estimators, max_estimators, 10):
    #         clf.set_params(n_estimators=i)
    #         clf.fit(X, y)
    #         oob_error = 1 - clf.oob_score_
    #         error_rate[label].append((i, oob_error))
    #
    # for label, clf_err in error_rate.items():
    #     xs, ys = zip(*clf_err)
    #     plt.plot(xs, ys, label=label)
    #
    # plt.xlim(min_estimators, max_estimators)
    # plt.xlabel("n_estimators")
    # plt.ylabel("OOB error rate")
    # plt.legend(loc="upper right")
    # plt.show()
    #
    # # Statistics:
    # y_pred = clf.predict(X_test)
    # labels = ['legit', 'dga']
    #
    # # Precision/Recall/f1-score.
    # cl_report = classification_report(y_test, y_pred, target_names=labels)
    # print(cl_report)
    #
    # # Evaluate classification accuracy by computing the confusion matrix.
    # # (In the matrix TRUE - dividend, PREDICATE - divisor)
    # cm = confusion_matrix(y_test, y_pred)
    # np.set_printoptions(precision=2)
    #
    # cm = cm.astype('float') / cm.sum(axis=1).T[:, np.newaxis]
    #
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title("Confusion matrix")
    # plt.colorbar()
    # plt.xticks([1, 0], labels, rotation=45)
    # plt.yticks([1, 0], labels)
    #
    # fmt = '.2f'
    # thresh = cm.max() / 2.
    # for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")
    #
    # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()


if __name__ == "__main__":
    train()
