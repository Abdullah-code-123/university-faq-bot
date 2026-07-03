# university-faq-bot
RAG-powered university FAQ bot built with FastAPI, Sentence Transformers, and Groq — answers student policy questions grounded in retrieved documents with injection detection and rate limiting
# University FAQ Bot — RAG-Powered Student Assistant

A FastAPI service that answers university policy questions 

grounded in real institutional documents, with semantic 

retrieval and structured responses.

## What it does

Students send a question (e.g. "what is the attendance policy?") 

and get back a structured answer grounded in retrieved university 

policy text — not model guesswork.

## Stack

- **FastAPI** — async REST API with Pydantic validation

- **Sentence Transformers** — local embeddings for semantic retrieval

- **Groq (LLaMA 3.3 70B)** — LLM inference via OpenAI-compatible SDK

- **RAG pipeline** — cosine similarity retrieval with relevance threshold

## Key engineering decisions

- Hard similarity threshold prevents hallucination when no 

  relevant policy exists — model explicitly flags low-confidence 

  answers instead of guessing

- Prompt injection detection blocks manipulation attempts 

  before reaching the LLM

- Rate limiting enforced in code (10 requests/60 seconds)

- Structured logging for production debugging

## How to run

1. Clone the repo

2. Install dependencies:

   pip install -r requirements.txt

3. Create a .env file:

   GROQ_API_KEY=your_key_here

4. Start the server:

   uvicorn weekproject2:app --reload

5. Open http://127.0.0.1:8000/docs

## Known limitations

- Knowledge base is 20 in-memory policy chunks

- Model may still answer from training data when 

  similarity score is below threshold

- Rate limiter resets on server restart
