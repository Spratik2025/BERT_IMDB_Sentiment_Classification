import os
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate


MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./models/distilbert-imdb"

print("Loading datasets...")

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]],
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df[["text", "label"]],
    preserve_index=False
)

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )


print("Tokenizing datasets...")

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True
)

tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True
)

tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_val = tokenized_val.remove_columns(["text"])

tokenized_train.set_format("torch")
tokenized_val.set_format("torch")


print("Loading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "NEGATIVE",
        1: "POSITIVE"
    },
    label2id={
        "NEGATIVE": 0,
        "POSITIVE": 1
    }
)


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = logits.argmax(axis=-1)

    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )

    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="binary"
    )

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=3,

    weight_decay=0.01,

    logging_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="f1",

    save_total_limit=2,

    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,

    tokenizer=tokenizer,

    compute_metrics=compute_metrics
)


print("\nStarting DistilBERT training...\n")

trainer.train()


print("\nFinal validation results:")
print(trainer.evaluate())


print("\nSaving model...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved to: {OUTPUT_DIR}")