from datasets import load_dataset
from sklearn.model_selection import train_test_split

print("Loading IMDb dataset...")

dataset = load_dataset("imdb")

train_data = dataset["train"]
test_data = dataset["test"]

# Convert to pandas
train_df = train_data.to_pandas()
test_df = test_data.to_pandas()

# Train / validation split
train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df["label"]
)

print("Dataset loaded successfully.")
print(f"Train samples      : {len(train_df)}")
print(f"Validation samples : {len(val_df)}")
print(f"Test samples       : {len(test_df)}")

train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("\nFiles created:")
print("train.csv")
print("validation.csv")
print("test.csv")