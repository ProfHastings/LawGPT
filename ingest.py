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
index = pinecone.GRPCIndex("justiz")

data = []
sources = []
print(f"Total files to process: {len(ps)}")
start_time = time.time()
for i, p in enumerate(ps):
    with open(p, 'rb') as f:
        text = f.read().decode('utf-8', errors='ignore')
        text = text.replace('\xa0', ' ')
        text = text.replace('\r\n', '\n')
        data.append(text)
    sources.append(p.stem)
    if (i + 1) % 1000 == 0: 
        elapsed_time = time.time() - start_time
        print(f"Processed {i+1} files in {elapsed_time:.2f} seconds.")


#text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=400)
text_splitter = NLTKTextSplitter(chunk_size=500
                                 #, chunk_overlap=200
                                 )
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    cleaned_splits = [split.replace('\n', ' ').replace('\t', ' ') for split in splits]
    docs.extend(cleaned_splits)
    metadatas.extend([{"source": f"{sources[i]}", "context": split} for split in cleaned_splits])

print(len(docs))
#for doc in docs:
#    print(doc, end = "\n\n")

batch_size = 25
splade_embeddings = []
print("encoding documents with splade_encoder...")
for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    splade_embeddings.extend(splade.encode_documents(batch))
    print(i+batch_size)

time.sleep(1)
print("Generating dense embeddings...")
start_time = time.time()
embeddings = []
batch_size = 25

for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    embeddings.extend(model.encode(batch, convert_to_tensor=True))
    splade_embeddings.extend(splade.encode_documents(batch))
    print(i+batch_size)
elapsed_time = time.time() - start_time
print(f"Generated embeddings for {len(docs)} documents in {elapsed_time:.2f} seconds.")

print(len(embeddings))

dense_embeddings = [emb.tolist() for emb in embeddings]

sparse_values = [{"indices": emb["indices"], "values": emb["values"]} for emb in splade_embeddings]

# Prepare the data for upsertion to Pinecone index
data_to_upsert = [{"id": f"{metadata['source']}_{i}", "values": dense_emb, "metadata": metadata, "sparse_values": sparse_emb} 
                  for i, (dense_emb, sparse_emb, metadata) in enumerate(zip(dense_embeddings, sparse_values, metadatas))]

# Upsert data into Pinecone index in chunks
for batch in chunks(data_to_upsert, batch_size=100):
    upsert_response = index.upsert(vectors=batch)
    if upsert_response.error:
        print(f'Error during upsert: {upsert_response.error}')
