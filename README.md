# Authorship Attribution Using Classification-Based Approach

Author: Catherine Yang

MSiA Text Analytics Independent Project

## Project Background

Authorship attribution (an application of “stylometry”) is the method of attributing text documents to authors. The main idea behind this discipline is to identify features of texts that can capture the syntactic and stylistic characteristics of an author’s writing. While this has many practical use cases, such as in forensics (i.e., identifying authors of disputed documents) and humanities research, the main motivation for pursuing this topic as a booklover was to understand how machine learning and deep learning models can be used to distinguish literary authors’ distinct styles of writing and compare the relative “uniqueness” of those styles. As such, this project comprises of two main objectives: 

1.	Assess classification model performance of literary texts using different types of feature generation and modeling approaches.
2.	Assess “uniqueness” of author’s writing style based on classification results for each author.

## Run API

To run the API, run the following command in the root of the repository:

```
python3 api.py
```

## Run Steps to Obtain Trained Model

### 1. Get Data

The data has already been imported from source and cleaned in the `\data` folder. To reimport and clean source text using an updated repertoire (i.e., updated `book_list.csv`), run the following command in the root of the repository:

```
python3 -m src.import_and_clean_data
```

### 2. Train Models

The Logistic Regression and Linear SVM models (with TF-IDF vectorization and Doc2Vec embeddings) have already been generated and saved in respective files in the `\models` folder. To re-train, run the following commands in the root of the repository:

```
python3 -m src.train_model_tfidf
python3 -m src.train_model_doc2vec
```



