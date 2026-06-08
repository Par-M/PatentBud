from sentence_transformers import SentenceTransformer
import json
import os

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

BASE_DIR = os.path.expanduser(
    "~/Desktop/patent-intelligence"
)

CHUNK_FOLDER = os.path.join(
    BASE_DIR,
    "chunks"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "embeddings",
    "embeddings.json"
)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

all_embeddings = []

for patent in os.listdir(CHUNK_FOLDER):

    patent_path = os.path.join(
        CHUNK_FOLDER,
        patent
    )

    if not os.path.isdir(patent_path):
        continue

    for file in os.listdir(patent_path):

        if not file.endswith(".txt"):
            continue

        filepath = os.path.join(
            patent_path,
            file
        )

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as f:

            content = f.read()

        embedding = model.encode(
            content
        ).tolist()

        all_embeddings.append({
            "patent": patent,
            "file": file,
            "text": content[:5000],
            "embedding": embedding
        })

print(
    f"Created {len(all_embeddings)} embeddings"
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_embeddings,
        f
    )