import json
import faiss
import numpy as np
import os

BASE_DIR = os.path.expanduser(
    "~/Desktop/patent-intelligence"
)

EMBEDDING_FILE = os.path.join(
    BASE_DIR,
    "embeddings",
    "embeddings.json"
)

with open(
    EMBEDDING_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)

vectors = np.array(
    [item["embedding"] for item in data]
).astype("float32")

dimension = vectors.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(vectors)

faiss.write_index(
    index,
    os.path.join(
        BASE_DIR,
        "embeddings",
        "patent_index.faiss"
    )
)

print(
    f"Stored {index.ntotal} vectors"
)