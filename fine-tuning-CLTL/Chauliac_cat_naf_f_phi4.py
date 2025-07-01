# -*- coding: utf-8 -*-
"""
Created on Wed May 21 21:05:17 2025

"""

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig
import numpy as np
import json
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.special import softmax

VALID_UD_TAGS = {
    "ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "VERB", "X", "SYM"
}

# Read sentences
with open("NAF_cat_chauliac.txt", "r") as f:
    sentences = [line.strip().split() for line in f]

# Read POS tags
df = pd.read_excel("NAF_cat_chauliac.xlsx")
all_tags = []
invalid_tags = set()

for tags_str in df["POS"]:
    tags = tags_str.split()
    all_tags.extend(tags)

# Check matching lengths
total_words = sum(len(sent) for sent in sentences)
assert total_words == len(all_tags), "Mismatch between tokens and tags!"

# Create samples
samples = []
index = 0
for words in sentences:
    tags = all_tags[index:index + len(words)]
    index += len(words)
    processed_tags = [(tag if tag in VALID_UD_TAGS else "UNK") for tag in tags]
    samples.append({"words": words, "tags": processed_tags})
    invalid_tags.update(tag for tag in tags if tag not in VALID_UD_TAGS)

if invalid_tags:
    print(f"⚠️ Invalid tags found: {invalid_tags}")

# Split into train/test
train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)

# Create tag2id
all_tags = {tag for sample in train_samples + test_samples for tag in sample["tags"]}
valid_ud_tags = sorted(VALID_UD_TAGS.intersection(all_tags))
tag2id = {tag: i for i, tag in enumerate(valid_ud_tags)}
id2tag = {i: tag for tag, i in tag2id.items()}

# Tokenizer and model
model_name="microsoft/phi-4"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="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", add_prefix_space=True)
tokenizer.pad_token = tokenizer.eos_token

# Dataset class
class POSTaggingDataset(Dataset):
    def __init__(self, samples, tokenizer, tag2id, max_length=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        words = self.samples[idx]["words"]
        tags = self.samples[idx]["tags"]

        encoding = self.tokenizer(
            words,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors="pt"
        )

        labels = []
        word_ids = encoding.word_ids(batch_index=0)
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(tag2id.get(tags[word_idx], -100))
                previous_word_idx = word_idx
            else:
                labels.append(-100)

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(labels)
        return encoding

# Create datasets
train_dataset = POSTaggingDataset(train_samples, tokenizer, tag2id)
test_dataset = POSTaggingDataset(test_samples, tokenizer, tag2id)

# Load model with LoRA
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(tag2id),
    device_map="auto",
    use_auth_token="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"
)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="TOKEN_CLS"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Metrics function with uncertainty
def compute_metrics(p):
    predictions, labels = p
    
    # Convert predictions to tensor and apply softmax to get probabilities
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    probabilities = F.softmax(predictions_tensor, dim=-1)
    
    # Get predicted classes
    predicted_classes = torch.argmax(probabilities, dim=-1).numpy()

    true_predictions = []
    true_labels = []

    # Uncertainty measures
    confidences = []
    uncertainties = []
    confidence_gaps = []

    for pred, label, prob in zip(predicted_classes, labels, probabilities.numpy()):
        for p, l, probs in zip(pred, label, prob):
            if l != -100:
                true_predictions.append(p)
                true_labels.append(l)

                # Softmax-based confidence and uncertainty
                max_prob = np.max(probs)  # Confidence
                confidences.append(max_prob)
                
                # Confidence gap between 1st and 2nd highest probabilities
                sorted_probs = np.partition(probs, -2)[-2:]
                confidence_gaps.append(sorted_probs[1] - sorted_probs[0])
                
                # Entropy for uncertainty
                uncertainties.append(entropy(probs))

    accuracy = accuracy_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions, average="macro", zero_division=0)
    recall = recall_score(true_labels, true_predictions, average="macro", zero_division=0)
    f1 = f1_score(true_labels, true_predictions, average="macro", zero_division=0)

    # Return metrics
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_confidence": np.mean(confidences),  # Mean confidence for predictions
        "mean_uncertainty": np.mean(uncertainties),  # Mean uncertainty (entropy)
        "mean_confidence_gap": np.mean(confidence_gaps),  # Mean confidence gap between top 2 classes
    }
os.environ.pop("RANK", None)
os.environ.pop("WORLD_SIZE", None)
os.environ.pop("LOCAL_RANK", None)

# Training arguments
training_args = TrainingArguments(
    output_dir="./f-lora-pos",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    optim="adamw_torch",
    logging_dir="./logs_f",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluation
test_results = trainer.predict(test_dataset)
predictions = test_results.predictions
labels = test_results.label_ids

true_predictions, true_labels = [], []
confidences = []
uncertainties = []
confidence_gaps = []

probabilities = softmax(predictions, axis=2)
predicted_classes = np.argmax(predictions, axis=2)

# Process predictions for uncertainty

for pred, label, prob in zip(predicted_classes, labels, probabilities):
    for p, l, probs in zip(pred, label, prob):
        if l != -100:
            true_predictions.append(p)
            true_labels.append(l)

            # Softmax-based confidence and uncertainty
            max_prob = max(probs)  # Confidence
            confidences.append(max_prob)
            confidence_gaps.append(np.partition(probs, -2)[-2] - max_prob)  # Confidence gap between 1st and 2nd
            uncertainties.append(entropy(probs))  # Entropy for uncertainty

# Generate classification report
report = classification_report(
    true_labels,
    true_predictions,
    labels=sorted(tag2id.values()),
    target_names=[id2tag[i] for i in sorted(tag2id.values())],
    digits=4
)

# Print the classification report
print("\nClassification Report\n" + "="*50)
print(report)

# Print the uncertainty measures
print("\nUncertainty Measures:")
print(f"Mean Confidence: {np.mean(confidences):.4f}")
print(f"Mean Uncertainty (Entropy): {np.mean(uncertainties):.4f}")
print(f"Mean Confidence Gap: {np.mean(confidence_gaps):.4f}")

# Save results and model
os.makedirs("chauliac_cat_naf", exist_ok=True)
model.save_pretrained("chauliac_cat_naf")
tokenizer.save_pretrained("chauliac_cat_naf")
with open("chauliac_cat_naf/tag2id_Chauliac_cat_naf.json", "w") as f:
    json.dump(tag2id, f)
with open("chauliac_cat_naf/classification_report_Chauliac_cat_naf.txt", "w") as f:
    f.write(report)

