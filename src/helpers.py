import pandas as pd
import numpy as np
import json
import string
import time
from itertools import chain

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def tokenize(text):
    """Returns a list of tokenized text."""

    tokens = map(lambda x: x.strip(string.punctuation), text.split())
    tokens = list(filter(None, tokens))
    return tokens


def tokenize_and_pos_tag(par):
    """Returns a list of lists with tokenized and POS tagged paragraphs."""

    tokens_pos_tagged = [
        nltk.pos_tag(text) for text in [tokenize(sent) for sent in sent_tokenize(par)]
    ]
    tokens_pos_tagged = list(chain.from_iterable(tokens_pos_tagged))

    return tokens_pos_tagged


def replace_proper_nouns(texts):
    """Returns a list of paragraphs where proper nouns are replaced with dummy text."""

    texts_pos = [tokenize_and_pos_tag(par) for par in texts]
    texts_pos = [
        [pair[1] if "NNP" in pair else pair[0] for pair in par] for par in texts_pos
    ]

    texts_replaced = [" ".join(par) for par in texts_pos]

    return texts_replaced


def vectorize_text_tfidf(texts, count_vect, tfidf_transformer):

    text_counts = count_vect.transform(texts)
    text_tfidf = tfidf_transformer.transform(text_counts)

    return text_tfidf


def vectorize_text_doc2vec(model, texts):

    pars_tokenized = [tokenize(text) for text in texts]
    pred_vector = [model.infer_vector(doc, steps=20) for doc in pars_tokenized]

    return pred_vector


def get_confusion_matrix(X, y_true, model):

    # Predicted label
    y_pred = model.predict(X)

    # Confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, normalize="true")
    df_cm = pd.DataFrame(cf_matrix, columns=np.unique(y_true), index=np.unique(y_true))
    df_cm.index.name = "Actual"
    df_cm.columns.name = "Predicted"

    return df_cm


def create_cf_viz(
    cf, model_name, colors="Blues", save=False, file_name=None,
):

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.set(font_scale=1.2)
    sns.heatmap(
        cf, cmap=colors, vmin=0, vmax=1, annot=True, annot_kws={"size": 14},
    )

    ax.set_title("Actual vs. Predicted Labels: " + model_name)
    plt.xticks(rotation=45)

    if save:
        plt.savefig(file_name, bbox_inches="tight")


def predict_text_print(text, label, model):

    predict_label = model.predict(text)

    output = pd.DataFrame(
        {"TEXT": text, "PREDICTED LABEL": predict_label, "ACTUAL LABEL": label}
    )
    parsed = json.loads(output.to_json(orient="index"))
    output_json = json.dumps(parsed, indent=4)

    print(output_json)


def print_metrics(y_pred, y_test):

    print("Accuracy is: %f" % accuracy_score(y_pred, y_test))
    print("Precision is: %f" % precision_score(y_pred, y_test, average="macro"))
    print("Recall is: %f" % recall_score(y_pred, y_test, average="macro"))
    print("F1 score is: %f" % f1_score(y_pred, y_test, average="macro"))
