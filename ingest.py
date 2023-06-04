from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import pickle
import torch
import time
import numpy as np
from pinecone_text.sparse import BM25Encoder
import pinecone

# Here we load in the data in the format that a directory of .txt files is in.
ps = list(Path("GmbHG/").glob("**/*.txt"))

# Initialize the SentenceTransformer model
model_name = 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
model = SentenceTransformer(model_name)

bm25_encoder = BM25Encoder()

index = pinecone.Index("justiz-9594bd4.svc.us-east-1-aws.pinecone.io")

data = []
sources = []
print(f"Total files to process: {len(ps)}")
start_time = time.time()
for i, p in enumerate(ps):
    with open(p, 'rb') as f:
        data.append(f.read().decode('utf-8', errors='ignore'))    
    sources.append(p.stem)
    if (i + 1) % 1000 == 0: # print every 1000 files
        elapsed_time = time.time() - start_time
        print(f"Processed {i+1} files in {elapsed_time:.2f} seconds.")


# Here we split the documents, as needed, into smaller chunks.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500)
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

print("fitting bm25_encoder...")
bm25_encoder.fit(docs)
bm25_encoder.dump("bm25_values.json")

# Generate document embeddings
print("Generating embeddings...")
start_time = time.time()
embeddings = []
batch_size = 1000  # You may need to adjust this depending on your GPU's memory capacity
for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    embeddings.extend(model(batch))
    print(i)
elapsed_time = time.time() - start_time
print(f"Generated embeddings for {len(docs)} documents in {elapsed_time:.2f} seconds.")

print(len(embeddings))

# Create a vector store from the embeddings and save it to disk
print("Creating vector store...")
start_time = time.time()
embeddings = list(zip(docs, embeddings))
#store = FAISS.from_embeddings(embeddings, model, metadatas=metadatas)
#faiss.write_index(store.index, "docs.index")
elapsed_time = time.time() - start_time
print(f"Created vector store in {elapsed_time:.2f} seconds.")

# Save the entire store
#with open("store.pkl", "wb") as f:
#    torch.save(store, f, pickle_protocol=4)
