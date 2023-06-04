from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import pickle
import torch
import time
from sentence_transformer_embeddings import SentenceTransformerEmbeddings
import numpy as np

# Here we load in the data in the format that a directory of .txt files is in.
ps = list(Path("database - Copy/").glob("**/*.txt"))

data = []
sources = []
print(f"Total files to process: {len(ps)}")
start_time = time.time()
for i, p in enumerate(ps):
    with open(p, 'rb') as f:
        data.append(f.read().decode('utf-8', errors='ignore'))
    sources.append(str(p))  # convert Path object to string
    if (i + 1) % 1000 == 0: # print every 1000 files
        elapsed_time = time.time() - start_time
        print(f"Processed {i+1} files in {elapsed_time:.2f} seconds.")

# Here we split the documents, as needed, into smaller chunks.
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

# Initialize the SentenceTransformer model
model_name = 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
st_model = SentenceTransformer(model_name)
model = SentenceTransformerEmbeddings(st_model)

# Save model
#st_model.to('cpu')
st_model.save("model_dir")

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
store = FAISS.from_embeddings(embeddings, model, metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
elapsed_time = time.time() - start_time
print(f"Created vector store in {elapsed_time:.2f} seconds.")

# Save the entire store
with open("store.pkl", "wb") as f:
    torch.save(store, f, pickle_protocol=4)
