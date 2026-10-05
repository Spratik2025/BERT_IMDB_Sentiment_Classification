BERT/DistilBERT Text Classification — IMDb Sentiment Analysis

Project Overview

This project implements text classification using a Transformer-based model. The objective is to classify movie reviews as either Positive or Negative.

Two approaches are implemented and compared:

TF-IDF + Logistic Regression — traditional machine-learning baseline

DistilBERT — Transformer-based deep-learning model

The project uses the IMDb Movie Review Dataset and fine-tunes distilbert-base-uncased for sentiment classification.

Objectives

Understand text classification using Transformers.

Preprocess and tokenize natural-language text.

Fine-tune a pretrained DistilBERT model.

Build a traditional ML baseline.

Compare Transformer and traditional ML performance.

Evaluate models using standard classification metrics.

Save and reload the trained Transformer model.

Perform sentiment prediction on new movie reviews.

Dataset

The project uses the IMDb dataset available through the Hugging Face datasets library.

The dataset contains 25,000 training reviews and 25,000 testing reviews with binary sentiment labels.

Label

Meaning

0

Negative

1

Positive

The original training set is divided into 80% training and 20% validation. The official test set is kept separate for final evaluation.

Models

1. TF-IDF + Logistic Regression

TF-IDF converts text into numerical feature vectors. These features are then passed to a Logistic Regression classifier to provide a traditional NLP baseline.

2. DistilBERT

The Transformer model used is:

distilbert-base-uncased

DistilBERT is a smaller and faster version of BERT while retaining much of BERT's language-understanding capability. The model is fine-tuned for binary sentiment classification.

Project Structure

BERT_IMDB_Sentiment_Classification/
│
├── 01_prepare_data.py
├── 02_baseline.py
├── 03_train_distilbert.py
├── 04_evaluate.py
├── 05_inference.py
├── requirements.txt
├── README.md
│
├── train.csv
├── validation.csv
├── test.csv
├── baseline_results.csv
├── distilbert_results.csv
│
└── models/
    └── distilbert-imdb/

Requirements

Recommended environment:

Python 3.13

Required libraries:

torch
torchvision
torchaudio
transformers
datasets
evaluate
accelerate
scikit-learn
pandas
numpy
matplotlib
seaborn

Installation

Create the virtual environment:

py -3.13 -m venv .venv

Activate it on Windows PowerShell:

.venv\Scripts\Activate.ps1

Upgrade pip:

python -m pip install --upgrade pip

Install dependencies:

pip install -r requirements.txt

For CPU-only PyTorch:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

How to Run

Run the scripts in this order.

Step 1 — Prepare Dataset

python 01_prepare_data.py

This downloads the IMDb dataset and creates:

train.csv
validation.csv
test.csv

Step 2 — Train Baseline Model

python 02_baseline.py

This trains TF-IDF + Logistic Regression and calculates Accuracy, Precision, Recall and F1-score.

Results are saved to:

baseline_results.csv

Step 3 — Fine-Tune DistilBERT

python 03_train_distilbert.py

The script tokenizes the reviews, loads pretrained DistilBERT, fine-tunes it, evaluates it on validation data, and saves the model.

Model location:

models/distilbert-imdb/

Step 4 — Evaluate DistilBERT

python 04_evaluate.py

This evaluates the saved model on the unseen IMDb test set.

Results are saved to:

distilbert_results.csv

Step 5 — Run Predictions

python 05_inference.py

The saved model is loaded and used to classify five new movie reviews.

Evaluation Metrics

Accuracy

Accuracy = Correct Predictions / Total Predictions

Precision

Measures how many reviews predicted as positive were actually positive.

Recall

Measures how many actual positive reviews were correctly identified.

F1-score

The harmonic mean of Precision and Recall:

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Model Comparison

After running both models, compare:

Model

Accuracy

Precision

Recall

F1

TF-IDF + Logistic Regression

—

—

—

—

DistilBERT

—

—

—

—

Use baseline_results.csv and distilbert_results.csv for the actual values.

Tokenization

DistilBERT uses:

AutoTokenizer.from_pretrained("distilbert-base-uncased")

Reviews are converted into token IDs before being passed to the model.

The project uses a maximum sequence length of 256. Long reviews are truncated and shorter reviews are padded.

Training Configuration

Model:
distilbert-base-uncased

Learning Rate:
2e-5

Batch Size:
8

Epochs:
3

Weight Decay:
0.01

Maximum Sequence Length:
256

These parameters can be adjusted depending on available hardware.

Hardware

The project can run on CPU or an NVIDIA GPU with CUDA. CPU training will take significantly longer than GPU training.

New Text / Unseen Data

For sentiment classification, the model can classify previously unseen text because it learned language representations during Transformer pretraining and sentiment patterns during fine-tuning.

Example Predictions

"This movie was absolutely fantastic and beautifully acted."
Expected: POSITIVE

"The movie was boring, predictable and poorly written."
Expected: NEGATIVE

"One of the best movies I have watched."
Expected: POSITIVE

"I completely disliked this movie."
Expected: NEGATIVE

Technologies Used

Programming

Python

Machine Learning

Scikit-learn

Logistic Regression

TF-IDF

Deep Learning

PyTorch

NLP / Transformers

Hugging Face Transformers

DistilBERT

Hugging Face Datasets

Data Processing

Pandas

NumPy

Evaluation

Scikit-learn Metrics

Hugging Face Evaluate

Common Issues

PyTorch DLL Error

If you receive:

OSError: [WinError 1114]

or:

Error loading c10.dll

make sure the correct PyTorch build and Microsoft Visual C++ Redistributable are installed.

For CPU:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Out of Memory

Reduce the training batch size:

per_device_train_batch_size=4

or:

per_device_train_batch_size=2

You can also reduce:

max_length=128

Model Saving

After training, the fine-tuned model is saved under:

models/distilbert-imdb/

The tokenizer is saved along with the model so that the same preprocessing can be used during inference.

Model Reloading

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    "./models/distilbert-imdb"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "./models/distilbert-imdb"
)

Learning Outcomes

After completing this project, you will understand:

Text classification

TF-IDF feature extraction

Logistic Regression

NLP preprocessing

Transformer architecture

BERT and DistilBERT

Tokenization

Attention masks

Fine-tuning pretrained models

Model evaluation

Accuracy, Precision, Recall and F1-score

Model saving and loading

Sentiment inference

Future Improvements

Fine-tune full BERT

Hyperparameter tuning

Learning-rate scheduling

Early stopping

Confusion matrix visualization

ROC-AUC analysis

Streamlit web application

REST API deployment

GPU/CUDA training

Experiment tracking with Weights & Biases

Deploy the model using Hugging Face Spaces

Author