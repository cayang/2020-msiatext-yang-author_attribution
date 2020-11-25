import pickle as pkl

import src.config as config
import src.helpers as helpers

# Load model in memory
def load_model():

    model_object_dict = {}

    for name, loc in config.DEPLOYED_ARTIFACTS_LOC.items():

        with open(loc, "rb") as file:
            model_object_dict[name] = pkl.load(file)

    return model_object_dict


model_object_dict = load_model()

# Generate predicted label
def vectorize_text(text, model_object_dict):

    if "tfidf_transformer" in model_object_dict.keys():
        count_vect = model_object_dict["count_vect"]
        tfidf_transformer = model_object_dict["tfidf_transformer"]
        text_vectorized = helpers.vectorize_text_tfidf(
            text, count_vect, tfidf_transformer
        )

    elif "doc2vec_model" in model_object_dict.keys():
        doc2vec_model = model_object_dict["doc2vec_model"]
        text_vectorized = helpers.vectorize_text_doc2vec(doc2vec_model, text)

    return text_vectorized


def generate_prediction(text_input):

    model = model_object_dict["model"]

    # Replace proper nouns
    text_input_replaced = helpers.replace_proper_nouns([text_input])

    # Vectorize the input text
    text_input_vectorized = vectorize_text(text_input_replaced, model_object_dict)

    # Generate prediction
    pred_label = model.predict(text_input_vectorized)

    return pred_label


if __name__ == "__main__":

    load_model()

    pred_label = generate_prediction(
        "It was the best of times, it was the worst of times."
    )
    print("Predicted label:", pred_label)

