from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch
import time
import numpy as np
from pinecone_text.sparse import BM25Encoder
import pinecone
import itertools

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.version())


def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

ps = list(Path("GmbHG/").glob("**/*.txt"))
model_name = 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(model_name)
model = model.to(device) # Move the model to the GPU

bm25_encoder = BM25Encoder(
    b=0.75,
    k1=1.2,
    lower_case=True,
    remove_punctuation=True,
    remove_stopwords=True,
    stem=False,
    language="german"
)


api_key = "2c3790ff-1d6a-48be-b101-1301723b6252"
env = "us-east-1-aws"

pinecone.init(api_key=api_key, environment=env)
index = pinecone.Index("justiz")

data = []
sources = []
print(f"Total files to process: {len(ps)}")
start_time = time.time()
for i, p in enumerate(ps):
    with open(p, 'rb') as f:
        data.append(f.read().decode('utf-8', errors='ignore'))    
    sources.append(p.stem)
    if (i + 1) % 1000 == 0: 
        elapsed_time = time.time() - start_time
        print(f"Processed {i+1} files in {elapsed_time:.2f} seconds.")

text_splitter = CharacterTextSplitter(chunk_size=2000, separator = "\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": f"{sources[i]}_{j}", "context": split} for j, split in enumerate(splits)])

print("fitting bm25_encoder...")
bm25_encoder.fit(docs)
bm25_encoder.dump("bm25_values.json")
#bm25_encoder = BM25Encoder().load("bm25_values.json")
bm25_embeddings = bm25_encoder.encode_documents(docs)

print("Generating embeddings...")
start_time = time.time()
embeddings = []
batch_size = 1000
for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
   # batch = [t.to(device) for t in batch] # Move the data to the GPU
    embeddings.extend(model.encode(batch, convert_to_tensor=True))

    print(i+batch_size)
elapsed_time = time.time() - start_time
print(f"Generated embeddings for {len(docs)} documents in {elapsed_time:.2f} seconds.")

print(len(embeddings))

dense_embeddings = [emb.tolist() for emb in embeddings]

# Assuming bm25_embeddings is a list of scipy csr_matrix
sparse_values = [{"indices": emb["indices"], "values": emb["values"]} for emb in bm25_embeddings]

# Prepare the data for upsertion to Pinecone index
data_to_upsert = [{"id": f"{metadata['source']}", "values": dense_emb, "metadata": metadata, "sparse_values": sparse_emb} 
                  for dense_emb, sparse_emb, metadata in zip(dense_embeddings, sparse_values, metadatas)]

# Upsert data into Pinecone index in chunks
for batch in chunks(data_to_upsert, batch_size=100):
    upsert_response = index.upsert(vectors=batch)
    if upsert_response.error:
        print(f'Error during upsert: {upsert_response.error}')
