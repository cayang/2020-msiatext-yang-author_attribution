import sys
import pathlib
import regex as re

# Filepaths
HOME = pathlib.Path(sys.path[0])
PATH_BOOK_LIST = HOME / "data" / "book_list.csv"
PATH_CLEAN_DATA = HOME / "data" / "clean_data.csv"
PATH_CLEAN_DATA_REPLACED = HOME / "data" / "clean_data_replaced.csv"
PATH_DOC2VEC = HOME / "models" / "model_dbow"
PATH_DOC2VEC_REPLACED = HOME / "models" / "model_dbow_replaced"
PATH_BERT_REPLACED = HOME / "models" / "fine_tuned_bert_v2"

# Deployed model
DEPLOYED_ARTIFACTS_LOC = {
    "model": str(HOME / "models" / "model_svm_replaced.pkl"),
    "count_vect": str(HOME / "models" / "countvect_svm_replaced.pkl"),
    "tfidf_transformer": str(HOME / "models" / "tfidftrans_svm_replaced.pkl"),
}

# Train settings
RANDOM_STATE = 414
CREATE_VIZ = True
LOAD_DOC2VEC_MODEL = True
LOAD_BERT = True

# Train model kwargs
lr_kwargs = {"solver": "liblinear"}
svm_kwargs = {"C": 0.5}
doc2vec_kwargs = {
    "dm": 0,
    "vector_size": 10,
    "alpha": 0.025,
    "negative": 2,
    "hs": 0,
    "min_count": 5,
    "sample": 0,
}

# Data Cleaning
REMOVE_WORDS = ["[deleted]"]
CHAPTER_REGEX = re.compile(r"(I.)[\s\w\\n]*(II).[\s\w\\n]*(III).[\s\w\\n]*")

# Flask configs
DEBUG = True
PORT = 5000
APP_NAME = "author_pred"
HOST = "127.0.0.1"
