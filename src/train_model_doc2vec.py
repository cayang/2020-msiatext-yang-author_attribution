import pandas as pd
import numpy as np
import scipy.stats
import string
import time
from tqdm import tqdm
from itertools import chain
import pickle as pkl

import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn import utils

import src.config as config
import src.helpers as helpers


def fit_model_doc2vec(train, test, epochs=20, kwargs=None):

    # Create tagged document
    train_tagged = create_tagged_doc(list(train["text"]), list(train["label"]))
    test_tagged = create_tagged_doc(list(test["text"]), list(test["label"]))

    # Build vocabulary
    model_dbow = Doc2Vec(train_tagged, **kwargs)

    # Train Doc2Vec model
    model_dbow.train(
        utils.shuffle([x for x in tqdm(train_tagged.values)]),
        total_examples=len(train_tagged.values),
        epochs=epochs,
    )

    return model_dbow, train_tagged, test_tagged


def create_tagged_doc(texts, labels):

    tagged_doc = [
        TaggedDocument(words=helpers.tokenize(texts[i]), tags=labels[i])
        for i in range(len(texts))
    ]

    return tagged_doc


def create_vec_for_learning(model, tagged_docs):

    targets, regressors = zip(
        *[(doc.tags, model.infer_vector(doc.words, steps=20)) for doc in tagged_docs]
    )

    return targets, regressors


def fit_model_classification(model, X_train, X_test, y_train, y_test):

    start_time = time.time()

    # Train model
    clf = model.fit(X_train, y_train)

    # Predict on test
    y_pred = clf.predict(X_test)

    end_time = time.time()

    print("Accuracy is: %f" % accuracy_score(y_pred, y_test))
    print("Precision is: %f" % precision_score(y_pred, y_test, average="macro"))
    print("Recall is: %f" % recall_score(y_pred, y_test, average="macro"))
    print("F1 score is: %f" % f1_score(y_pred, y_test, average="macro"))
    print("Execution time: %0.2fs" % (end_time - start_time))

    return clf


def get_doc2vec_model(
    df, load_doc2vec=True, save_doc2vec=False, file_name=None, kwargs=None
):

    # Split into train / test set
    train, test = train_test_split(df, test_size=0.2, random_state=config.RANDOM_STATE)

    # Fit Doc2Vec model to get embeddings
    if load_doc2vec:
        model_dbow = Doc2Vec.load(file_name)
        train_tagged = create_tagged_doc(list(train["text"]), list(train["label"]))
        test_tagged = create_tagged_doc(list(test["text"]), list(test["label"]))

    else:
        model_dbow, train_tagged, test_tagged = fit_model_doc2vec(
            train, test, kwargs=kwargs
        )

    if save_doc2vec:
        model_dbow.save(file_name)

    return model_dbow, train_tagged, test_tagged


def get_trained_model(
    train_tagged,
    test_tagged,
    doc2vec_model,
    model_type="LogisticRegression",
    kwargs=None,
):

    # Use traditional ML to perform classification
    y_train, X_train = create_vec_for_learning(doc2vec_model, train_tagged)
    y_test, X_test = create_vec_for_learning(doc2vec_model, test_tagged)

    # Define model and train
    if model_type == "SVM":
        model = LinearSVC(multi_class="ovr", random_state=config.RANDOM_STATE, **kwargs)
    else:
        model = LogisticRegression(
            multi_class="ovr", random_state=config.RANDOM_STATE, **kwargs
        )

    clf = fit_model_classification(model, X_train, X_test, y_train, y_test)

    return clf


def output_viz(X, y, doc2vec_model, model, model_name, file_name, colors="Blues"):

    # Vectorize texts
    X_vectorized = helpers.vectorize_text_doc2vec(doc2vec_model, X)

    # Get confusion matrix
    cf = helpers.get_confusion_matrix(X_vectorized, y, model)

    # Create visualization
    helpers.create_cf_viz(cf, model_name, colors, save=True, file_name=file_name)


if __name__ == "__main__":

    # Read in data
    df = pd.read_csv(config.PATH_CLEAN_DATA, encoding="utf_8")
    df_replaced = pd.read_csv(config.PATH_CLEAN_DATA_REPLACED, encoding="utf_8")

    # Get Doc2vec embeddings
    model_dbow, train_tagged, test_tagged = get_doc2vec_model(
        df,
        load_doc2vec=config.LOAD_DOC2VEC_MODEL,
        file_name=str(config.PATH_DOC2VEC),
        kwargs=config.doc2vec_kwargs,
    )
    (
        model_dbow_replaced,
        train_tagged_replaced,
        test_tagged_replaced,
    ) = get_doc2vec_model(
        df_replaced,
        load_doc2vec=config.LOAD_DOC2VEC_MODEL,
        file_name=str(config.PATH_DOC2VEC_REPLACED),
        kwargs=config.doc2vec_kwargs,
    )

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
    clf_lr = get_trained_model(
        train_tagged,
        test_tagged,
        model_dbow,
        model_type="LogisticRegression",
        kwargs=config.lr_kwargs,
    )
    clf_lr_replaced = get_trained_model(
        train_tagged_replaced,
        test_tagged_replaced,
        model_dbow_replaced,
        model_type="LogisticRegression",
        kwargs=config.lr_kwargs,
    )
    clf_svm = get_trained_model(
        train_tagged,
        test_tagged,
        model_dbow,
        model_type="SVM",
        kwargs=config.svm_kwargs,
    )
    clf_svm_replaced = get_trained_model(
        train_tagged_replaced,
        test_tagged_replaced,
        model_dbow_replaced,
        model_type="SVM",
        kwargs=config.svm_kwargs,
    )

    # Save models and objects
    file_model_lr_replaced = str(config.HOME) + "/models/model_dbow_lr_replaced.pkl"
    file_model_svm_replaced = str(config.HOME) + "/models/model_dbow_svm_replaced.pkl"

    pkl.dump(clf_lr_replaced, open(file_model_lr_replaced, "wb"))
    pkl.dump(clf_svm_replaced, open(file_model_svm_replaced, "wb"))

    # Generate confusion matrices and graphs
    if config.CREATE_VIZ:
        viz_path_lr = str(config.HOME) + "/output/lr_dbow.png"
        viz_path_lr_replaced = str(config.HOME) + "/output/lr_dbow_replaced.png"
        viz_path_svm = str(config.HOME) + "/output/svm_dbow.png"
        viz_path_svm_replaced = str(config.HOME) + "/output/svm_dbow_replaced.png"

        output_viz(
            X_test,
            y_test,
            model_dbow,
            clf_lr,
            "Logistic Regression",
            viz_path_lr,
            colors="Blues",
        )

        output_viz(
            X_test_replaced,
            y_test_replaced,
            model_dbow_replaced,
            clf_lr_replaced,
            "Logistic Regression (Proper Nouns Replaced)",
            viz_path_lr_replaced,
            colors="Greens",
        )

        output_viz(
            X_test,
            y_test,
            model_dbow,
            clf_svm,
            "Linear SVM",
            viz_path_svm,
            colors="Blues",
        )

        output_viz(
            X_test_replaced,
            y_test_replaced,
            model_dbow_replaced,
            clf_svm_replaced,
            "Linear SVM (Proper Nouns Replaced)",
            viz_path_svm_replaced,
            colors="Greens",
        )

