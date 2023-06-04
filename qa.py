"""Ask a question to the notion database."""

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import argparse
import faiss
from langchain.vectorstores import FAISS
import pickle
from sentence_transformer_embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()

# Load model
model_dir = os.path.abspath("model_dir")
st_model = SentenceTransformer.load(model_dir)
model = SentenceTransformerEmbeddings(st_model)


# Load index
index = faiss.read_index("docs.index")

# Load metadata
with open("store.pkl", "rb") as f:
    store = torch.load('store.pkl',map_location ='cpu')

# Recreate the vector store
#store = FAISS(index=index, model=model, metadatas=store_metadata.metadatas)

chain = RetrievalQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=store.as_retriever())
result = chain({"question": args.question})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
