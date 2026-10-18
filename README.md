# Dialogue Summarizer (T5 + FastAPI)

A web app that summarizes conversations using a fine-tuned T5 model.

## Features
- Fine-tuned T5 model for dialogue summarization
- FastAPI backend with a `/summarize` REST endpoint
- Simple HTML/JS frontend

## Tech Stack
Python, PyTorch, Hugging Face Transformers, FastAPI, HTML/CSS/JavaScript

## Run locally
1. Clone the repo and install dependencies:
```
   pip install -r requirements.txt
```
2. Download the model from Hugging Face: https://huggingface.co/YOUR_USERNAME/t5-dialogue-summarizer
   and put the files in a folder named `saved_summary_model/`
3. Start the server:
```
   uvicorn app:app
```
4. Open http://127.0.0.1:8000

## API
`POST /summarize` with `{"dialogue": "..."}` returns `{"summary": "..."}`