from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import NLTKTextSplitter
from sentence_transformers import SentenceTransformer
import torch
import time
import numpy as np
import pinecone
import itertools
from pinecone_text.sparse import SpladeEncoder
from langchain.embeddings import OpenAIEmbeddings

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.version())

def chunks(iterable, batch_size=10):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def process_chunk_upsert(index, batch_to_upsert, max_retries=10):
    for attempt in range(max_retries):
        try:
            upsert_response = index.upsert(vectors=batch_to_upsert)
            if upsert_response.error:
                raise Exception(f'Error during upsert: {upsert_response.error}')
            break
        except Exception:
            if attempt < max_retries - 1:  # i.e. if it's not the last attempt
                continue
            else:
                print("FAILURE")
                return batch_to_upsert
    return []

def process_embedding(doc, metadata):
    try:
        dense_embedding = openai_embedding_model.embed_documents([doc])[0]
        sparse_embedding = splade.encode_documents([doc])[0]
        item_to_upsert = {"id": f"{metadata['long_source']}_{j}", "values": dense_embedding, "metadata": metadata, "sparse_values": sparse_embedding} 
        return item_to_upsert, None
    except Exception as e:
        print(f"Embedding failed with error: {str(e)}")
        return None, doc

ps = list(Path("AngG/").glob("**/*.txt"))
model_name = 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(model_name, device=device)
splade = SpladeEncoder(device=device)
openai_embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", max_retries=50)

api_key = "953b2be8-0621-42a1-99db-8480079a9e23"
env = "eu-west4-gcp"

pinecone.init(api_key=api_key, environment=env)
index = pinecone.Index("justiz-openai")

print(f"Total files to process: {len(ps)}")
start_time = time.time()

embedding_counter = 0
MAX_RETRIES = 100
failed_chunks = []

for i, p in enumerate(ps):
    with open(p, 'rb') as f:
        text = f.read().decode('utf-8', errors='ignore')
        text = text.replace('\xa0', ' ').replace('\r\n', '\n')
        lines = text.split("\n")
        short_source = ""  # default if no Geschäftszahl found
        for j, line in enumerate(lines):
            if "Geschäftszahl" in line and j < len(lines) - 1:
                short_source = lines[j + 1]
                break
        source = {"long_source": p.stem, "short_source": short_source}

    text_splitter = CharacterTextSplitter(chunk_size=1000, separator="\n")
    splits = text_splitter.split_text(text)

    cleaned_splits = [split.replace('\n', ' ').replace('\t', ' ') for split in splits]
    metadatas = [{"long_source": source["long_source"], "short_source": source["short_source"], "context": split} for j, split in enumerate(cleaned_splits)]

    batch_to_upsert = []
    for j, (doc, metadata) in enumerate(zip(cleaned_splits, metadatas)):
        embedding_counter += 1
        if(embedding_counter < 20000):
            continue
        item_to_upsert, failed_doc = process_embedding(doc, metadata)
        if item_to_upsert is not None:
            batch_to_upsert.append(item_to_upsert)
        else:
            failed_chunks.append([failed_doc])

        if len(batch_to_upsert) >= 10:
            failed_chunk = process_chunk_upsert(index, batch_to_upsert, MAX_RETRIES)
            if failed_chunk:
                failed_chunks.append(failed_chunk)
            batch_to_upsert = []

    # Upsert remaining items in the batch, if any
    if batch_to_upsert:
        failed_chunk = process_chunk_upsert(index, batch_to_upsert, MAX_RETRIES)
        if failed_chunk:
            failed_chunks.append(failed_chunk)

    elapsed_time = time.time() - start_time
    print(f"Processed {i} files in {elapsed_time:.2f} seconds.")
    print(f"Processed {embedding_counter} embeddings in {elapsed_time:.2f} seconds.")

print("Failed chunks:")
for chunk in failed_chunks:
    for item in chunk:
        print(item['id'])
print("Created Embeddings:")
print(embedding_counter)