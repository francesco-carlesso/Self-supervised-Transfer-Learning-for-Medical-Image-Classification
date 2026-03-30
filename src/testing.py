import gc
import torch
import numpy as np
from medmnist import BreastMNIST, PneumoniaMNIST
from transformers import AutoModelForImageClassification, AutoImageProcessor, Trainer
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.special import softmax
from preprocessing import load_dataset

def Testing(dataset, model_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForImageClassification.from_pretrained(model_path).to(device)
    processor = AutoImageProcessor.from_pretrained(model_path)

    # Preprocessing (BEiT requires RGB. MedMNIST is grayscale)
    def preprocess_images(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        inputs = processor(images, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs

    dataset.set_transform(preprocess_images)

    trainer = Trainer(
        model=model,
        processing_class=processor
    )

    prediction_output = trainer.predict(dataset["test"])
    probs = softmax(np.array(prediction_output.predictions), axis=1)
    y_score = probs[:, 1]
    pred_labels = prediction_output.predictions.argmax(axis=1)
    true_labels = prediction_output.label_ids

    accuracy = accuracy_score(true_labels, pred_labels)
    auc = roc_auc_score(true_labels, y_score)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")

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

for conf in configs:
    name = f"{conf['class'].__name__}_{conf['size']}_{'balanced' if conf['balanced'] else ''}"
    print(f"\n# {'='*40}\n# EVALUATING EXPERIMENT: {name}\n# {'='*40}")

    model_checkpoint = f"models/{name}" 
    
    current_data = load_dataset(conf['class'], conf['size'], conf['balanced'], conf['alpha'])

    Testing(
        dataset=current_data,
        model_path=model_checkpoint,
    )

    del current_data
    gc.collect()