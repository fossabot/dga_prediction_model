"""
Training model for detect DGA domains using machine learning
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from scipy.stats import entropy

from config import training_data


def train_model():
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
    # "(x,y) n" mean that "(row, column) value"
    # Value - the number of times a ngram appeared in the domains
    # represented by the row of the matrix
    ngram_matrix = vectorizer.fit_transform(training_data['legit']['domain'])

    #
    #
    ngram_counts = np.log10(ngram_matrix.sum(axis=0).getA1())

    #
    #
    all_data_dict['occur_ngrams'] = ngram_counts * vectorizer.transform(all_data_dict['domain']).T

    print(all_data_dict)
    # x = all_data_dict.as_matrix(['legit_ngrams'])
    # y = np.array(all_data_dict['type'].tolist())2
    # clf = RandomForestClassifier(n_estimators=10)
    # clf = clf.fit(x, y)


if __name__ == "__main__":
    train_model()
