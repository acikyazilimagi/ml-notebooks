
import argparse
import numpy as np
import os
import torch as th

from collections import OrderedDict
from huggingface_hub import login
from sklearn.metrics import classification_report
from transformers import (AdamW, AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments, EarlyStoppingCallback,
                          get_cosine_schedule_with_warmup)

from utils.generic_utils import set_seed_everywhere, select_thresholds, compute_f1
from utils.dataset_utils import prep_datasets
from utils.training_utils import ImbalancedTrainer, compute_class_weights


LABEL_IDX2NAME = OrderedDict([
        (0, 'Lojistik'),
        (1, 'Elektrik Kaynagi'),
        (2, 'Arama Ekipmani'),
        (3, 'Cenaze'),
        (4, 'Giysi'),
        (5, 'Enkaz Kaldirma'),
        (6, 'Isinma'),
        (7, 'Barınma'),
        (8, 'Tuvalet'),
        (9, 'Su'),
        (10, 'Yemek'),
        (11, 'Saglik'),
        (12, 'Alakasiz')])

os.environ["WANDB_DISABLED"] = "true"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


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


def main():
    # Define argpars for training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=1, help="Number of trials to run with different seeds")
    parser.add_argument("--model_name", type=str, default="dbmdz/bert-base-turkish-uncased",
                        help="Name or path of the model to use. For example, could be"
                             "<path-to-BERT-finetuned-for-MLM-on-unlabelled-tweets>")
    parser.add_argument("--output_dir", type=str, default="./output-intent")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--layerwise_LR_decay_rate", type=float, default=0.8)
    args = parser.parse_args()

    login(token=args.hf_token)
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
    data_collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=100)
    train_ds, val_ds, test_ds, mlb_labels = prep_datasets(
        tokenizer,
        labelidx2name=LABEL_IDX2NAME,
        path="deprem-private/intent-v13")

    f1s = []
    for i in range(args.n_seeds):
        set_seed_everywhere(i)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=len(LABEL_IDX2NAME), problem_type="multi_label_classification")

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            report_to=None,
            num_train_epochs=15,
            metric_for_best_model="macro f1",
            load_best_model_at_end=True,
            group_by_length=True
        )
        optimizer, scheduler = get_llrd_optimizer_scheduler(
            model,
            learning_rate=5e-5,
            weight_decay=0.01,  # Weight decay defined here instead of training_args
            layerwise_learning_rate_decay=args.layerwise_LR_decay_rate)

        trainer = ImbalancedTrainer(
            class_weights=compute_class_weights(mlb_labels),
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
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
            target_names=LABEL_IDX2NAME.keys(), digits=3)
        print(report)

    print("Mean F1: {:.2f}, Std F1: {:.2f}".format(np.mean(f1s) * 100, np.std(f1s) * 100))


if __name__ == '__main__':
    main()
