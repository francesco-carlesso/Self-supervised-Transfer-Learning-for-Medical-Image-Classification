import os
import shutil
import random
import gc
import torch
import numpy as np
from medmnist import BreastMNIST, PneumoniaMNIST
from transformers import AutoModelForImageClassification, AutoImageProcessor, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.special import softmax
from preprocessing import load_dataset

def set_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def beit_freezer(model):
    for name, param in model.named_parameters():
        if not name.startswith("classifier") \
            and not name.startswith("beit.pooler")\
            and not name.startswith("beit.encoder.layer.23")\
            and not name.startswith("beit.encoder.layer.22")\
            and not name.startswith("beit.encoder.layer.21")\
            and not name.startswith("beit.encoder.layer.20")\
            and not name.startswith("beit.encoder.layer.19"):
            param.requires_grad = False

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = softmax(logits, axis=1)[:, 1]
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, probs)
    return {"accuracy": accuracy, "auc": auc}

def Training(dataset, model_path, output_dir, batch_size, weight_decay, freezer):

    set_reproducibility()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForImageClassification.from_pretrained(model_path, num_labels=2, ignore_mismatched_sizes=True).to(device)
    processor = AutoImageProcessor.from_pretrained(model_path)

    freezer(model)

    # Preprocessing (BEiT requires RGB. MedMNIST is grayscale)
    def preprocess_images(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        inputs = processor(images, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs

    dataset.set_transform(preprocess_images)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=weight_decay,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=1,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            shutil.rmtree(item_path)

    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

configs = [
    {"class": BreastMNIST, "size": 28, "balanced": False, "alpha": 0},
    {"class": BreastMNIST, "size": 28, "balanced": True,  "alpha": 0.5},
    {"class": BreastMNIST, "size": 224, "balanced": False, "alpha": 0},
    {"class": BreastMNIST, "size": 224, "balanced": True,  "alpha": 0.5},
    {"class": PneumoniaMNIST, "size": 28, "balanced": False, "alpha": 0},
    {"class": PneumoniaMNIST, "size": 28, "balanced": True,  "alpha": 0},
    {"class": PneumoniaMNIST, "size": 224, "balanced": False, "alpha": 0},
    {"class": PneumoniaMNIST, "size": 224, "balanced": True,  "alpha": 0},
]

model_checkpoint = "microsoft/beit-large-patch16-224-pt22k"

for conf in configs:
    name = f"{conf['class'].__name__}_{conf['size']}_{'balanced' if conf['balanced'] else ''}"
    print(f"\n# {'='*40}\n# EXPERIMENT: {name}\n# {'='*40}")
    
    current_data = load_dataset(conf['class'], conf['size'], conf['balanced'], conf['alpha'])
    
    Training(
        dataset=current_data,
        model_path=model_checkpoint,
        output_dir=f"models/{name}",
        batch_size=32,
        weight_decay=0.1,
        freezer=beit_freezer
    )

    del current_data
    gc.collect()