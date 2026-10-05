import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


MODEL_PATH = "./models/distilbert-imdb"

print("Loading test dataset...")

test_df = pd.read_csv("test.csv")

test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]],
    preserve_index=False
)

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )


test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.remove_columns(["text"])

test_dataset.set_format("torch")


print("Loading fine-tuned model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)


trainer = Trainer(
    model=model,
    tokenizer=tokenizer
)


print("\nRunning predictions...")

predictions = trainer.predict(test_dataset)

logits = predictions.predictions
y_true = predictions.label_ids

y_pred = logits.argmax(axis=-1)


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)


print("\n==============================")
print("DistilBERT TEST RESULTS")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Negative", "Positive"]
    )
)


results = pd.DataFrame({
    "Model": ["DistilBERT"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1]
})

results.to_csv(
    "distilbert_results.csv",
    index=False
)

print("\nResults saved to distilbert_results.csv")