import pandas as pd
import urllib
import requests
import regex as re
from bs4 import BeautifulSoup
import gutenberg_cleaner as gclean

from nltk.corpus import stopwords

import src.config as config
import src.helpers as helpers


def clean_paragraphs(par):
    """Clean paragraphs in each book"""

    # Replace special characters
    par = par.replace("\n", " ")
    par = par.replace("\\", "")

    # Remove leading whitespace
    par = par.lstrip()

    return par


if __name__ == "__main__":

    # Get list of books and their links
    df = pd.read_csv(config.PATH_BOOK_LIST)

    # Set up vectors to store paragraphs (data) and labels (authors)
    X = []
    y = []
    stop_words = set(stopwords.words("english"))

    for _, row in df.iterrows():
        label = row.get("author")
        target_url = row.get("link")

        # Get text
        soup_html = urllib.request.urlopen(target_url).read()
        text = str(BeautifulSoup(soup_html, features="lxml"))
        text_clean = gclean.super_cleaner(text)

        # Separate into paragraphs
        paragraphs = text_clean.split("\n\n")
        paragraphs = [item for item in paragraphs if item not in config.REMOVE_WORDS]
        paragraphs = [
            par for par in paragraphs if re.search(config.CHAPTER_REGEX, par) is None
        ]

        # Clean paragraphs
        paragraphs = list(map(lambda x: clean_paragraphs(x), paragraphs))

        # Append to data
        X.extend(paragraphs)
        y.extend([label] * len(paragraphs))

        # Replace proper nouns in paragraphs
        X_replaced = helpers.replace_proper_nouns(X)

        # Create an output set
        out_data = pd.DataFrame({"text": X, "label": y})
        out_data.to_csv(str(config.HOME) + "/data/clean_data.csv", index=False)

        out_data_replaced = pd.DataFrame({"text": X_replaced, "label": y})
        out_data_replaced.to_csv(
            str(config.HOME) + "/data/clean_data_replaced.csv", index=False
        )

