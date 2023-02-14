
import argparse
import numpy as np
import pandas as pd
import torch as th

from datasets import load_dataset
from huggingface_hub import login
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from unicode_tr import unicode_tr
from transformers import (AdamW, AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, EarlyStoppingCallback,
                          get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup)

from ..utils import set_seed_everywhere, select_thresholds, compute_f1


LABEL_NAMES = [
    'Alakasiz', 'Barinma', 'Elektronik',
    'Giysi', 'Kurtarma', 'Lojistik', 'Saglik',
    'Su', 'Guvenlik', 'Yemek']


def get_optimizer_grouped_parameters(
    model, model_type,
    learning_rate, weight_decay,
    layerwise_learning_rate_decay
):
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    # initialize lrs for every layer
    num_layers = model.config.num_hidden_layers
    layers = [getattr(model, model_type).embeddings] + list(getattr(model, model_type).encoder.layer)
    layers.reverse()
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters


def get_llrd_optimizer_scheduler(model, learning_rate=1e-5, weight_decay=0.01, layerwise_learning_rate_decay=0.95):
    grouped_optimizer_params = get_optimizer_grouped_parameters(
        model, 'bert',
        learning_rate, weight_decay,
        layerwise_learning_rate_decay
    )
    optimizer = AdamW(
        grouped_optimizer_params,
        lr=learning_rate,
        eps=1e-6,
        correct_bias=True
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=15
    )
    # Note: linear schedule fails to converge for unknown reasons.

    return optimizer, scheduler


def prep_datasets(tokenizer,
                  path="deprem-private/deprem_intent_classification",
                  name="intent_multilabel"):
    intent = load_dataset(path, name, use_auth_token=True)
    print(intent["train"], intent["test"])

    for instance in intent["train"]:
        print(unicode_tr(instance["text_cleaned"]).lower())
        break

    df_train = pd.DataFrame().from_records(list(intent["train"]))
    df_test = pd.DataFrame().from_records(list(intent["test"]))

    label_col = "labels"
    text_col = "text_cleaned"
    df_train[text_col] = df_train[text_col].apply(lambda x: unicode_tr(x).lower())
    df_test[text_col] = df_test[text_col].apply(lambda x: unicode_tr(x).lower())

    df_train = df_train[df_train[label_col].notnull()].reset_index(drop=True)
    df_test = df_test[df_test[label_col].notnull()].reset_index(drop=True)

    df_train.labels.apply(lambda x: len(x))

    labels = set()
    for label in df_train.labels.values:
        labels.update({l for l in label})

    labels = list(sorted(labels))

    stratify = [np.random.choice(ls) for ls in df_train[label_col].values]
    df_train["stratify"] = stratify
    df_train, df_valid = train_test_split(df_train, test_size=0.1, stratify=df_train["stratify"], random_state=42)

    train_ds = IntentDataset(df_train, tokenizer, len(labels), label_col, text_col)
    val_ds = IntentDataset(df_valid, tokenizer, len(labels), label_col, text_col)
    test_ds = IntentDataset(df_test, tokenizer, len(labels), label_col, text_col)

    return train_ds, val_ds, test_ds


class IntentDataset(th.utils.data.Dataset):

    def __init__(self, df, tokenizer, num_classes=-1, label_col="labels", text_col="text_cleaned"):
        self.df = df
        self.tokenizer = tokenizer
        self.num_classes = num_classes
        self.label_col = label_col
        self.text_col = text_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text, label = row.text, self._encode_label(row[self.label_col])
        encoding = self.tokenizer(text, max_length=64, padding="max_length", truncation=True)
        encoding = {key: th.tensor(val) for key, val in encoding.items()}
        encoding[self.label_col] = th.tensor(label)
        return dict(encoding)

    def _encode_label(self, labels):
        encoded_labels = np.zeros(self.num_classes)
        for label in labels:
            encoded_labels[label] = 1.0
        return encoded_labels


def main():
    # Define argpars for training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=1, help="Number of trials to run with different seeds")
    parser.add_argument("--model_name", type=str, default="dbmdz/bert-base-turkish-uncased",
                        help="Name or path of the model to use. For example, could be"
                             "<path-to-BERT-finetuned-for-MLM-on-unlabelled-tweets>")
    parser.add_argument("--output_dir", type=str, default="./output-intent")
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--layerwise_LR_decay_rate", type=float, default=0.8)
    args = parser.parse_args()

    login(token=args.hf_token)
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
    train_ds, val_ds, test_ds = prep_datasets(tokenizer)

    f1s = []
    for i in range(args.n_seeds):
        set_seed_everywhere(i)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=len(LABEL_NAMES), problem_type="multi_label_classification")

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            report_to=None,
            num_train_epochs=15,
            metric_for_best_model="f1",
            load_best_model_at_end=True,
        )
        optimizer, scheduler = get_llrd_optimizer_scheduler(
            model,
            learning_rate=5e-5,
            weight_decay=0.01,  # Weight decay defined here instead of training_args
            layerwise_learning_rate_decay=args.layerwise_LR_decay_rate)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_f1,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            optimizers=(optimizer, scheduler)
        )
        trainer.train()

        # Choose the best thresholds per label using train+val data
        train_preds = trainer.predict(train_ds)
        val_preds = trainer.predict(val_ds)
        thresholds = select_thresholds(
            np.concatenate([train_preds.label_ids, val_preds.label_ids]),
            np.concatenate([train_preds.predictions, val_preds.predictions])
        )
        # Evaluate on test data
        test_preds = trainer.predict(test_ds)
        f1 = compute_f1((test_preds.predictions, test_preds.label_ids), thresholds=thresholds)
        f1s.append(f1["f1"])
        report = classification_report(
            test_preds.label_ids.astype(int),
            (th.sigmoid(th.from_numpy(test_preds.predictions)).numpy() > thresholds).astype(int),
            target_names=LABEL_NAMES, digits=3)
        print(report)

    print("Mean F1: {:.2f}, Std F1: {:.2f}".format(np.mean(f1s) * 100, np.std(f1s) * 100))


if __name__ == '__main__':
    main()
