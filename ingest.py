from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import NLTKTextSplitter
from sentence_transformers import SentenceTransformer
import torch
import time
import numpy as np
import pinecone
import itertools
from pinecone_text.sparse import SpladeEncoder

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.version())


def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

ps = list(Path("AngG/").glob("**/*.txt"))
model_name = 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(model_name, device=device)
splade = SpladeEncoder(device=device)

api_key = "2c3790ff-1d6a-48be-b101-1301723b6252"
env = "us-east-1-aws"

pinecone.init(api_key=api_key, environment=env)
index = pinecone.Index("justiz")

data_to_upsert = []
print(f"Total files to process: {len(ps)}")
start_time = time.time()

for i, p in enumerate(ps):
    with open(p, 'rb') as f:
        lines = f.read().decode('utf-8', errors='ignore').split("\n")
        text = ' '.join(lines).replace('\xa0', ' ').replace('\r\n', '\n')
        short_source = ""  # default if no Geschäftszahl found
        for j, line in enumerate(lines):
            if "Geschäftszahl" in line and j < len(lines) - 1:
                short_source = lines[j + 1]
                break
        source = {"long_source": p.stem, "short_source": short_source}

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=400)
    text_splitter = NLTKTextSplitter(chunk_size=500)
    splits = text_splitter.split_text(text)
    cleaned_splits = [split.replace('\n', ' ').replace('\t', ' ') for split in splits]
    metadatas = [{"long_source": source["long_source"], "short_source": source["short_source"], "context": split} for j, split in enumerate(cleaned_splits)]

    for j, (doc, metadata) in enumerate(zip(cleaned_splits, metadatas)):
        dense_embedding = model.encode([doc], convert_to_tensor=True)[0].tolist()
        sparse_embedding = splade.encode_documents([doc])[0]
        item_to_upsert = {"id": f"{metadata['long_source']}_{j}", "values": dense_embedding, "metadata": metadata, "sparse_values": sparse_embedding} 
        data_to_upsert.append(item_to_upsert)

    if (i + 1) % 1000 == 0:
        elapsed_time = time.time() - start_time
        print(f"Processed {i+1} files in {elapsed_time:.2f} seconds.")

for batch in chunks(data_to_upsert, batch_size=100):
    upsert_response = index.upsert(vectors=batch)
    if upsert_response.error:
        print(f'Error during upsert: {upsert_response.error}')
