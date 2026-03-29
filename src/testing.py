import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import Dataset as HFDataset, DatasetDict
from medmnist import BreastMNIST, PneumoniaMNIST
from transformers import AutoModelForImageClassification, AutoImageProcessor, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import random
from collections import Counter
from sklearn.utils import resample
import datasets
from scipy.special import softmax
from preprocessing import load_dataset

def preprocess_images(examples):
    images = [img.convert("RGB") for img in examples["image"]]
    inputs = processor(images, return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs

dataset.set_transform(preprocess_images)

print(f"--- Final Evaluation on Test Set ({output_dir}) ---")
predictions = trainer.predict(dataset["test"])
probs = softmax(np.array(predictions.predictions), axis=1)
y_score = probs[:, 1]
pred_labels = predictions.predictions.argmax(axis=1)
true_labels = predictions.label_ids

accuracy = accuracy_score(true_labels, pred_labels)
auc = roc_auc_score(true_labels, y_score)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")