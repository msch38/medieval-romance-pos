# -*- coding: utf-8 -*-
"""
Created on Wed May 21 21:05:17 2025

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    Gemma3ForCausalLM
)
from peft import get_peft_model, LoraConfig, TaskType
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

# Define valid UD tags
VALID_UD_TAGS = {
    "ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "VERB", "X", "SYM"
}

# Read sentences
with open("Chauliac_cat_naf.txt", "r") as f:
    sentences = [line.strip().split() for line in f]

# Read POS tags
df = pd.read_excel("Chauliac_cat_naf.xlsx")
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

# Create tag2id mapping
all_tags_set = {tag for sample in train_samples + test_samples for tag in sample["tags"]}
valid_ud_tags = sorted(VALID_UD_TAGS.intersection(all_tags_set))
tag2id = {tag: i for i, tag in enumerate(valid_ud_tags)}
id2tag = {i: tag for tag, i in tag2id.items()}
num_labels = len(tag2id)

print(f"Number of classes: {num_labels}")
print(f"Classes: {list(tag2id.keys())}")

# Model setup
hugging_face_model_id = "google/gemma-3-12b-it"
gpu_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    hugging_face_model_id, 
    use_auth_token="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", 
    add_prefix_space=True
)
tokenizer.pad_token = tokenizer.eos_token

# Load the base Gemma3 model first
base_model = Gemma3ForCausalLM.from_pretrained(
    hugging_face_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    #attn_implementation='eager',
    use_auth_token="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"
)

# Replace the lm_head with classification head directly on the base model
base_model.lm_head = nn.Linear(
    base_model.config.hidden_size,
    num_labels,
    #bias=False,
    dtype=torch.bfloat16
).to(gpu_device)

# Initialize the new classification head
#nn.init.normal_(base_model.lm_head.weight, std=0.02)

# Add config attribute for num_labels (needed for some training utilities)
base_model.config.num_labels = num_labels

# Custom forward method to handle token classification
def custom_forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
    # Ensure we get hidden states if not already specified
    if 'output_hidden_states' not in kwargs:
        kwargs['output_hidden_states'] = True
    
    # Call the original forward method
    outputs = self._original_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **kwargs
    )
    
    # Get logits (now from classification head)
    logits = outputs.logits
    
    loss = None
    if labels is not None:
        # Compute token classification loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    
    # Return with modified loss
    return type(outputs)(
        loss=loss,
        logits=logits,
        past_key_values=getattr(outputs, 'past_key_values', None),
        hidden_states=getattr(outputs, 'hidden_states', None),
        attentions=getattr(outputs, 'attentions', None)
    )

# Store original forward and replace with custom one
base_model._original_forward = base_model.forward
base_model.forward = custom_forward.__get__(base_model, base_model.__class__)

# Use the modified base model directly
model = base_model



# Apply LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="TOKEN_CLS",  # Keep as CAUSAL_LM since we're using Gemma3ForCausalLM
    modules_to_save=["lm_head"]  # Ensure the classification head is saved
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

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
                labels.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                # First subtoken of a word gets the label
                if word_idx < len(tags):
                    labels.append(self.tag2id.get(tags[word_idx], -100))
                else:
                    labels.append(-100)
                previous_word_idx = word_idx
            else:
                # Subsequent subtokens of the same word get -100
                labels.append(-100)

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoding

# Create datasets
train_dataset = POSTaggingDataset(train_samples, tokenizer, tag2id)
test_dataset = POSTaggingDataset(test_samples, tokenizer, tag2id)

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
                max_prob = np.max(probs)
                confidences.append(max_prob)
                
                # Confidence gap between 1st and 2nd highest probabilities
                sorted_probs = np.sort(probs)
                if len(sorted_probs) >= 2:
                    confidence_gaps.append(sorted_probs[-1] - sorted_probs[-2])
                else:
                    confidence_gaps.append(max_prob)
                
                # Entropy for uncertainty
                uncertainties.append(entropy(probs + 1e-8))  # Add small epsilon to avoid log(0)

    if len(true_predictions) == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    accuracy = accuracy_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions, average="macro", zero_division=0)
    recall = recall_score(true_labels, true_predictions, average="macro", zero_division=0)
    f1 = f1_score(true_labels, true_predictions, average="macro", zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_confidence": np.mean(confidences) if confidences else 0,
        "mean_uncertainty": np.mean(uncertainties) if uncertainties else 0,
        "mean_confidence_gap": np.mean(confidence_gaps) if confidence_gaps else 0,
    }

# Clean environment variables
for env_var in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]:
    os.environ.pop(env_var, None)

# Training arguments
training_args = TrainingArguments(
    output_dir="./f-lora-pos-gemma3",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=8,  # Reduced further for 12B model
    per_device_eval_batch_size=8,
    #gradient_accumulation_steps=2,  # Effective batch size = 4
    num_train_epochs=10,
    weight_decay=0.01,
    optim="adamw_torch",
    logging_dir="./logs_f_gemma3",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    #save_total_limit=1,
    #dataloader_pin_memory=False,
    #gradient_checkpointing=True,
    #remove_unused_columns=False,
    #fp16=False,  # Use bfloat16 instead
    #bf16=True,
    #dataloader_num_workers=0,  # Avoid multiprocessing issues
)

# Standard trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
print("Starting training...")
trainer.train()

# Evaluation
print("Starting evaluation...")
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
            max_prob = np.max(probs)
            confidences.append(max_prob)
            
            sorted_probs = np.sort(probs)
            if len(sorted_probs) >= 2:
                confidence_gaps.append(sorted_probs[-1] - sorted_probs[-2])
            else:
                confidence_gaps.append(max_prob)
                
            uncertainties.append(entropy(probs + 1e-8))

# Generate classification report
if len(true_predictions) > 0:
    report = classification_report(
        true_labels,
        true_predictions,
        labels=list(range(len(tag2id))),
        target_names=[id2tag[i] for i in range(len(tag2id))],
        digits=4
    )
    
    # Print results
    print("\nClassification Report\n" + "="*50)
    print(report)
    
    print("\nUncertainty Measures:")
    print(f"Mean Confidence: {np.mean(confidences):.4f}")
    print(f"Mean Uncertainty (Entropy): {np.mean(uncertainties):.4f}")
    print(f"Mean Confidence Gap: {np.mean(confidence_gaps):.4f}")
else:
    print("No valid predictions found!")
    report = "No valid predictions found!"

# Save results and model
os.makedirs("chauliac_cat_naf_gemma3", exist_ok=True)
model.save_pretrained("chauliac_cat_naf_gemma3")
tokenizer.save_pretrained("chauliac_cat_naf_gemma3")

with open("chauliac_cat_naf_gemma3/tag2id_Chauliac_cat_naf_f_g3.json", "w") as f:
    json.dump(tag2id, f)
    
with open("chauliac_cat_naf_gemma3/classification_report_Chauliac_cat_naf_f_g3.txt", "w") as f:
    f.write(report)

