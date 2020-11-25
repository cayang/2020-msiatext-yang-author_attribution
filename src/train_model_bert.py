# Inspiration source for code: https://towardsdatascience.com/multi-class-text-classification-with-deep-learning-using-bert-b59ca2f5c613

import pandas as pd
import numpy as np
import random
import torch
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

import src.config as config
import src.helpers as helpers

# Set parameters
batch_size = 3
epochs = 5

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device")

seed_val = config.RANDOM_STATE
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def prep_data(df):

    # Encode labels
    possible_labels = df.label.unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index

    # Replace text labels with encoded value
    df["label"] = df.label.replace(label_dict)

    return df, label_dict


def get_confusion_matrix(y_pred, y_true):

    # Confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, normalize="true")
    df_cm = pd.DataFrame(cf_matrix, columns=np.unique(y_true), index=np.unique(y_true))
    df_cm.index.name = "Actual"
    df_cm.columns.name = "Predicted"

    return df_cm


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average="weighted")


def accuracy_per_class(preds, labels, label_dict):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.vectorize(label_dict_inverse.get)(
        np.argmax(preds, axis=1).flatten()
    )
    labels_flat = np.vectorize(label_dict_inverse.get)(labels.flatten())

    helpers.print_metrics(preds_flat, labels_flat)
    print("\n")

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f"Class: {label}")
        print(f"Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n")

    cf = get_confusion_matrix(preds_flat, labels_flat)
    helpers.create_cf_viz(
        cf, "BERT (Proper Nouns Replaced) - 10K Training Samples", colors="Greens"
    )


def evaluate(model, dataloader_val):

    model.to(device)
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:

        batch = tuple(b.to(device) for b in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
        }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


def main():

    # Only read in data for replaced
    df = pd.read_csv(config.PATH_CLEAN_DATA_REPLACED, encoding="utf_8")
    df, label_dict = prep_data(df)

    # Split train vs. val
    X_train, X_val, y_train, y_val = train_test_split(
        df.index.values,
        df.label.values,
        train_size=80000,
        test_size=20000,
        random_state=414,
        stratify=df.label.values,
    )

    # Label train vs. val
    df["data_type"] = ["not_set"] * df.shape[0]

    df.loc[X_train, "data_type"] = "train"
    df.loc[X_val, "data_type"] = "val"

    # Encode data
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type == "train"].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=25,
        return_tensors="pt",
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type == "val"].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=25,
        return_tensors="pt",
    )

    input_ids_train = encoded_data_train["input_ids"]
    attention_masks_train = encoded_data_train["attention_mask"]
    labels_train = torch.tensor(df[df.data_type == "train"].label.values)

    input_ids_val = encoded_data_val["input_ids"]
    attention_masks_val = encoded_data_val["attention_mask"]
    labels_val = torch.tensor(df[df.data_type == "val"].label.values)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    # Load pretrained BERT model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_dict),
        output_attentions=False,
        output_hidden_states=False,
    )

    # Train and validation data loader
    dataloader_train = DataLoader(
        dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size
    )

    dataloader_validation = DataLoader(
        dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size
    )

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train) * epochs
    )

    # Train
    for epoch in tqdm(range(1, epochs + 1)):

        model.to(device)
        model.train()

        loss_train_total = 0

        progress_bar = tqdm(
            dataloader_train,
            desc="Epoch {:1d}".format(epoch),
            leave=False,
            disable=False,
        )

        for batch in progress_bar:

            model.zero_grad()

            batch = tuple(b.to(device) for b in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
            }

            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix(
                {"training_loss": "{:.3f}".format(loss.item() / len(batch))}
            )

        torch.save(
            model.state_dict(), f"./models/finetuned_BERT_epoch_{epoch}.model",
        )

        tqdm.write(f"\nEpoch {epoch}")

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f"Training loss: {loss_train_avg}")

        val_loss, predictions, true_vals = evaluate(model, dataloader_validation)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f"Validation loss: {val_loss}")
        tqdm.write(f"F1 Score (Weighted): {val_f1}")


if __name__ == "__main__":

    main()

