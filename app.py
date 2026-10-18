import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = FastAPI(title="text summarization", description="Text summarization API using T5", version="1.0")

model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

def clean_text(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()

def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_text(dialogue)
    inputs = tokenizer(
        dialogue,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(targets[0], skip_special_tokens=True)

@app.post("/summarize")
def summarize(dialogue_input: DialogueInput):
    return {"summary": summarize_dialogue(dialogue_input.dialogue)}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")