import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_PATH = "./models/distilbert-imdb"

print("Loading saved model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

examples = [
    "This movie was absolutely fantastic. The acting was brilliant and the story was amazing.",

    "I hated this movie. It was boring, slow and poorly written.",

    "The movie was okay, but the story could have been much better.",

    "One of the best films I have ever watched. Highly recommended!",

    "Terrible movie. I wasted two hours watching it."
]


def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    label = model.config.id2label[
        predicted_class
    ]

    return label, confidence


print("\n==============================")
print("SENTIMENT PREDICTIONS")
print("==============================")

for i, text in enumerate(examples, start=1):
    label, confidence = predict_sentiment(text)

    print(f"\nExample {i}")
    print("Review    :", text)
    print("Sentiment :", label)
    print("Confidence:", f"{confidence:.2%}")