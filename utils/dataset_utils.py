
import numpy as np
import pandas as pd
import torch as th

from datasets import load_dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from unicode_tr import unicode_tr


def prep_datasets(tokenizer, labelidx2name, path, label_col="label", text_col="image_url"):
    intent = load_dataset(path, use_auth_token=True)
    print(intent["train"], intent["test"])

    for instance in intent["train"]:
        print(unicode_tr(instance["image_url"]).lower())
        break

    df_train = pd.DataFrame().from_records(list(intent["train"]))
    df_test = pd.DataFrame().from_records(list(intent["test"]))

    df_train[text_col] = df_train[text_col].apply(lambda x: unicode_tr(x).lower())
    df_test[text_col] = df_test[text_col].apply(lambda x: unicode_tr(x).lower())

    # Next, we remove the rows that have no labels
    df_train = df_train[df_train[label_col].notnull()].reset_index(drop=True)
    df_test = df_test[df_test[label_col].notnull()].reset_index(drop=True)

    # df_train.labels.apply(lambda x: len(x))
    #
    # labels = set()
    # for label in df_train.labels.values:
    #     labels.update({l for l in label})
    #
    # name2ix = {v: k for k, v in labelidx2name.items()}
    # labels = name2ix.keys()

    mlb = MultiLabelBinarizer(classes=list(labelidx2name.values()))
    mlb_labels = mlb.fit_transform(df_train.label.tolist())

    cv = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(cv.split(df_train.index.tolist(), mlb_labels)):
        df_train.loc[val_idx, 'kfold'] = int(fold)

    df_train, df_val = df_train[df_train['kfold'] != 0], df_train[df_train['kfold'] == 0]

    train_ds = IntentDataset(df_train, tokenizer, labelidx2name, label_col, text_col)
    val_ds = IntentDataset(df_val, tokenizer, labelidx2name, label_col, text_col)
    test_ds = IntentDataset(df_test, tokenizer, labelidx2name, label_col, text_col)

    return train_ds, val_ds, test_ds, mlb_labels


class IntentDataset(th.utils.data.Dataset):

    def __init__(self, df, tokenizer, labelidx2name, label_col="label", text_col="image_url"):
        self.df = df
        self.tokenizer = tokenizer
        self.labelidx2name = labelidx2name
        self.name2ix = {v: k for k, v in labelidx2name.items()}
        self.num_classes = len(labelidx2name)
        self.label_col = label_col
        self.text_col = text_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text, label = row.image_url, self._encode_label(row[self.label_col])
        encoding = self.tokenizer(text, max_length=100, padding="max_length", truncation=True)
        encoding = {key: th.tensor(val, dtype=th.int64) for key, val in encoding.items()}
        encoding[self.label_col] = th.tensor(label, dtype=th.float32)
        return dict(encoding)

    def _encode_label(self, labels):
        encoded_labels = np.zeros(self.num_classes)
        for label in labels:
            encoded_labels[self.name2ix[label]] = 1.0
        return encoded_labels
