from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import argparse
import torch
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import SpladeEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
api_key = "2c3790ff-1d6a-48be-b101-1301723b6252"
env = "us-east-1-aws"
pinecone.init(api_key=api_key, environment=env)
index = pinecone.Index("justiz")

parser = argparse.ArgumentParser(description='Ask a question to the legal database.')
parser.add_argument('question', type=str, help='The question to ask the legal database')
args = parser.parse_args()

# Load
splade = SpladeEncoder(device=device)
model_name = 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
embeddings = HuggingFaceEmbeddings(model_name=model_name, device=device)

retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=splade, index=index, top_k=4)

chain = RetrievalQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(temperature=0, model="gpt-4"), retriever=retriever, verbose = True)
result = chain({"question": args.question})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
