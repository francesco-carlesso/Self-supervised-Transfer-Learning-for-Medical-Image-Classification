import io
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from transformers import AutoModelForImageClassification, AutoImageProcessor
from contextlib import asynccontextmanager

MODEL_PATHS = {
    "breast": "models/BreastMNIST_224_balanced",
    "pneumonia": "models/PneumoniaMNIST_224_balanced"
}

models = {}
processors = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):

    print(f"Loading models to {device}...")

    for task, path in MODEL_PATHS.items():
        try:
            models[task] = AutoModelForImageClassification.from_pretrained(path).to(device)
            processors[task] = AutoImageProcessor.from_pretrained(path)
            models[task].eval()
            print(f"Successfully loaded {task} model.")

        except Exception as e:
            print(f"Failed to load {task} model from {path}. Error: {e}")

    yield

    models.clear()
    processors.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="MedMNIST API", lifespan=lifespan)

@app.post("/predict")
async def predict(
    task: str = Form(..., description="Specify the model to use: 'breast' or 'pneumonia'"),
    file: UploadFile = File(..., description="The image file to classify")
):

    task = task.lower()
    if task not in models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid task '{task}'. Available tasks are: {list(models.keys())}"
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file format. Please upload a valid image.")

    processor = processors[task]
    model = models[task]
    
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    
    if predicted_class == 0 and task == 'breast':
        predicted_class = 'Positive'
    elif predicted_class == 1 and task == 'breast':
        predicted_class = 'Negative'
    elif predicted_class == 0 and task == 'pneumonia':
        predicted_class = 'Negative'
    elif predicted_class == 1 and task == 'pneumonia':
        predicted_class = 'Positive'

    return {
        "task": task,
        "predicted_class": predicted_class, # Positives: Cancer=0 Pneumonia=1
        "confidence": round(confidence, 4),
        "probabilities": [round(p, 4) for p in probs[0].tolist()]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)