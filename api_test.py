from langchain.retrievers import PineconeHybridSearchRetriever
import torch
import pinecone
import argparse
from pinecone_text.sparse import SpladeEncoder
from langchain.embeddings import HuggingFaceEmbeddings
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#Pinecone
api_key = "2c3790ff-1d6a-48be-b101-1301723b6252"
env = "us-east-1-aws"
pinecone.init(api_key=api_key, environment=env)
index = pinecone.GRPCIndex("justiz")

# Loading models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splade = SpladeEncoder(device=device)
model_name = 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
embeddings = HuggingFaceEmbeddings(model_name=model_name)

retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=splade, index=index, top_k=50, alpha = 0.3) #lower alpha - more sparse

results = retriever.get_relevant_documents("Regelt das AngG Kündigungstermine?")

for result in results:
    print(result)