from sentence_transformers import SentenceTransformer
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

import json
import faiss
import numpy as np
import os

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

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
    10
)

context = ""

for idx in I[0]:

    chunk = metadata[idx]

    context += f"""

PATENT: {chunk['patent']}

FILE: {chunk['file']}

{chunk['text']}

"""

response = client.chat.completions.create(

    model="llama-3.3-70b-versatile",

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
    ]
)

print("\n")
print(response.choices[0].message.content)