from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np
import os

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
    "r"
) as f:

    data = json.load(f)

index = faiss.read_index(
    os.path.join(
        BASE_DIR,
        "embeddings",
        "patent_index.faiss"
    )
)

query = input(
    "Search patents: "
)

query_vector = model.encode(
    query
).astype("float32")

D, I = index.search(
    np.array([query_vector]),
    5
)

print("\nTop Results:\n")

for idx in I[0]:

    result = data[idx]

    print(
        f"\nPatent: {result['patent']}"
    )

    print(
        f"File: {result['file']}"
    )

    print(
        result['text'][:500]
    )