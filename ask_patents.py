import json
import os

import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

OLLAMA_LLM_MODEL = "gemma:7b"

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

BASE_DIR = os.path.expanduser(
    "~/Desktop/patent-intelligence"
)

with open(
    os.path.join(
        BASE_DIR,
        "embeddings",
        "embeddings.json"
    ),
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)

index = faiss.read_index(
    os.path.join(
        BASE_DIR,
        "embeddings",
        "patent_index.faiss"
    )
)

query = input(
    "\nAsk a patent question:"
)

query_embedding = model.encode(
    query
).astype("float32")

D, I = index.search(
    np.array([query_embedding]),
    8
)

context = ""

for idx in I[0]:

    chunk = metadata[idx]

    context += f"""

PATENT: {chunk['patent']}

FILE: {chunk['file']}

{chunk['text']}

"""

response = ollama.chat(
    model=OLLAMA_LLM_MODEL,
    messages=[
        {
            "role": "system",
            "content":
            """
            You are an expert patent analyst.

            Your job is to identify:
            - patent overlap
            - prior art
            - competitive threats
            - infringement risk
            - opportunities for differentiation
            """
        },

        {
            "role": "user",
            "content":
            f"""
            Startup Context:

            Verity is an AI platform helping users improve:
            - workplace communication
            - social confidence
            - professional conversations
            - cultural fluency
            - communication coaching

            Question:

            {query}

            Patent Context:

            {context}

            Return:

            1. Executive Summary

            2. Relevant Patents

            3. Overlap With Verity

            4. Risk Score (1-10)

            5. Why It Matters

            6. Opportunities To Differentiate

            7. Recommended Founder Action
            """
        }
    ],
    options={"temperature": 0.2},
)

print("\n")
print(response["message"]["content"])
