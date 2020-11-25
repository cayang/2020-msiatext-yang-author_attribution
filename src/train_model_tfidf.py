import pandas as pd
import numpy as np
import scipy.stats
import string
import time
from itertools import chain
import pickle as pkl


import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

import src.config as config
import src.helpers as helpers


def fit_model_tfidf(model, X_train, X_test, y_train, y_test, use_bigrams=False):

    start_time = time.time()

    # Vectorize
    if use_bigrams == True:
        count_vect = CountVectorizer(ngram_range=(2, 2), max_df=0.7, min_df=5)
    else:
        count_vect = CountVectorizer(max_df=0.7, min_df=5)
    tfidf_transformer = TfidfTransformer()

    X_train_counts = count_vect.fit_transform(X_train)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Train model
    clf = model.fit(X_train_tfidf, y_train)

    # Apply vectorization on test set
    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    # Predict on test
    y_pred = clf.predict(X_test_tfidf)

    end_time = time.time()

    print("Accuracy is: %f" % accuracy_score(y_pred, y_test))
    print("Precision is: %f" % precision_score(y_pred, y_test, average="macro"))
    print("Recall is: %f" % recall_score(y_pred, y_test, average="macro"))
    print("F1 score is: %f" % f1_score(y_pred, y_test, average="macro"))
    print("Execution time: %0.2fs" % (end_time - start_time))

    return clf, count_vect, tfidf_transformer


def get_trained_model(
    df, model_type="LogisticRegression", use_bigrams=False, kwargs=None
):

    # Get regressors and labels
    texts = list(df["text"])
    labels = list(df["label"])

    # Split into train / test set
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=config.RANDOM_STATE
    )

    # Define model and train
    if model_type == "SVM":
        model = LinearSVC(multi_class="ovr", random_state=config.RANDOM_STATE, **kwargs)
    else:
        model = LogisticRegression(
            multi_class="ovr", random_state=config.RANDOM_STATE, **kwargs
        )

    clf, count_vect, tfidf_transformer = fit_model_tfidf(
        model, X_train, X_test, y_train, y_test, use_bigrams=use_bigrams
    )

    return clf, count_vect, tfidf_transformer


def output_viz(
    X, y, model, count_vect, tfidf_transformer, model_name, file_name, colors="Blues"
):

    # Vectorize texts
    X_vectorized = helpers.vectorize_text_tfidf(X, count_vect, tfidf_transformer)

    # Get confusion matrix
    cf = helpers.get_confusion_matrix(X_vectorized, y, model)

    # Create visualization
    helpers.create_cf_viz(cf, model_name, colors, save=True, file_name=file_name)


if __name__ == "__main__":

    # Read in data
    df = pd.read_csv(config.PATH_CLEAN_DATA, encoding="utf_8")
    df_replaced = pd.read_csv(config.PATH_CLEAN_DATA_REPLACED, encoding="utf_8")

    # Split into train / test set
    _, X_test, _, y_test = train_test_split(
        list(df["text"]),
        list(df["label"]),
        test_size=0.2,
        random_state=config.RANDOM_STATE,
    )
    _, X_test_replaced, _, y_test_replaced = train_test_split(
        list(df_replaced["text"]),
        list(df_replaced["label"]),
        test_size=0.2,
        random_state=config.RANDOM_STATE,
    )

    # Get trained model objects
    clf_lr, count_vect_lr, tfidf_transformer_lr = get_trained_model(
        df, model_type="LogisticRegression", kwargs=config.lr_kwargs
    )
    (
        clf_lr_replaced,
        count_vect_lr_replaced,
        tfidf_transformer_lr_replaced,
    ) = get_trained_model(
        df_replaced, model_type="LogisticRegression", kwargs=config.lr_kwargs
    )
    clf_svm, count_vect_svm, tfidf_transformer_svm = get_trained_model(
        df, model_type="SVM", kwargs=config.svm_kwargs
    )
    (
        clf_svm_replaced,
        count_vect_svm_replaced,
        tfidf_transformer_svm_replaced,
    ) = get_trained_model(df_replaced, model_type="SVM", kwargs=config.svm_kwargs)

    # Save models and objects
    file_model_lr_replaced = str(config.HOME) + "/models/model_lr_replaced.pkl"
    file_model_svm_replaced = str(config.HOME) + "/models/model_svm_replaced.pkl"
    file_countvect_lr_replaced = str(config.HOME) + "/models/countvect_lr_replaced.pkl"
    file_countvect_svm_replaced = (
        str(config.HOME) + "/models/countvect_svm_replaced.pkl"
    )
    file_tfidftrans_lr_replaced = (
        str(config.HOME) + "/models/tfidftrans_lr_replaced.pkl"
    )
    file_tfidftrans_svm_replaced = (
        str(config.HOME) + "/models/tfidftrans_svm_replaced.pkl"
    )

    pkl.dump(clf_lr_replaced, open(file_model_lr_replaced, "wb"))
    pkl.dump(clf_svm_replaced, open(file_model_svm_replaced, "wb"))
    pkl.dump(count_vect_lr_replaced, open(file_countvect_lr_replaced, "wb"))
    pkl.dump(count_vect_svm_replaced, open(file_countvect_svm_replaced, "wb"))
    pkl.dump(tfidf_transformer_lr_replaced, open(file_tfidftrans_lr_replaced, "wb"))
    pkl.dump(tfidf_transformer_svm_replaced, open(file_tfidftrans_svm_replaced, "wb"))

    # Generate confusion matrices and graphs
    if config.CREATE_VIZ:
        viz_path_lr = str(config.HOME) + "/output/lr.png"
        viz_path_lr_replaced = str(config.HOME) + "/output/lr_replaced.png"
        viz_path_svm = str(config.HOME) + "/output/svm.png"
        viz_path_svm_replaced = str(config.HOME) + "/output/svm_replaced.png"

        output_viz(
            X_test,
            y_test,
            clf_lr,
            count_vect_lr,
            tfidf_transformer_lr,
            "Logistic Regression",
            viz_path_lr,
            colors="Blues",
        )

        output_viz(
            X_test_replaced,
            y_test_replaced,
            clf_lr_replaced,
            count_vect_lr_replaced,
            tfidf_transformer_lr_replaced,
            "Logistic Regression (Proper Nouns Replaced)",
            viz_path_lr_replaced,
            colors="Greens",
        )

        output_viz(
            X_test,
            y_test,
            clf_svm,
            count_vect_svm,
            tfidf_transformer_svm,
            "Linear SVM",
            viz_path_svm,
            colors="Blues",
        )

        output_viz(
            X_test_replaced,
            y_test_replaced,
            clf_svm_replaced,
            count_vect_svm_replaced,
            tfidf_transformer_svm_replaced,
            "Linear SVM (Proper Nouns Replaced)",
            viz_path_svm_replaced,
            colors="Greens",
        )
