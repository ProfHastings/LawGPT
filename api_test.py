from langchain.retrievers import PineconeHybridSearchRetriever
import torch
import pinecone
import argparse
from pinecone_text.sparse import SpladeEncoder
from langchain.embeddings import HuggingFaceEmbeddings
import os
from tiktoken import Tokenizer, TokenizerHelper

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#Pinecone
api_key = "2c3790ff-1d6a-48be-b101-1301723b6252"
env = "us-east-1-aws"
pinecone.init(api_key=api_key, environment=env)
index = pinecone.Index("justiz")

# Loading models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splade = SpladeEncoder(device=device)
model_name = 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
embeddings = HuggingFaceEmbeddings(model_name=model_name)

retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=splade, index=index, top_k=40, alpha = 0.3) #lower alpha - more sparse

question = "Beurteile den folgenden Fall: Adam arbeitet in der Autowerkstätte von Fred als angestellter Mechaniker. Ohne Wissen von Fred arbeitet Adam nachts des öfteren in einem call center für Milchprodukte. Manchmal arbeitet er dort die ganze Nacht und kommt nicht zum Schlafen. Als Fred davon erfärt, ist er empört. Kann er Adam diese Tätigkeit neben seinem Dienstverhältnis mit Fred verbieten?"
results = retriever.get_relevant_documents(question)

contents = [doc.page_content for doc in results]
#sources = [result.metadata['source'] for result in results]
#unique_sources = list(dict.fromkeys(sources).keys())

#for source in unique_sources:
#    print(source)
#print(len(unique_sources))

#for result in results:
#    print(result)
#for result in results:
#    print(f"Inhalt: {result.page_content}")
#    print(f"Quelle: {result.metadata['source']}")




# initialize tokenizer
helper = TokenizerHelper()
tokenizer = Tokenizer.from_file(helper.get_model_file("gpt4"))

token_count = 0
max_token_limit = 100
gpt4prompt = "Du bist ein erfahrener Anwalt. Deine Aufgabe ist es die folgende Frage zu beantworten: INSERT QUESTION HERE  Um die Frage zu beantworten hast du die folgenden Entscheidungen des Österreichischen  Obersten Gerichtshofes zur Verfügung:"

for result in results:
    new_text = f"Inhalt: {result.page_content}\nQuelle: {result.metadata['source']}\n"
    new_tokens = list(tokenizer.tokenize(new_text))
    new_token_count = len(new_tokens)

    if token_count + new_token_count > max_token_limit:
        break

    gpt4prompt += new_text
    token_count += new_token_count

print(gpt4prompt)
