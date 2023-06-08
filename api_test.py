from langchain import PromptTemplate
from langchain.retrievers import PineconeHybridSearchRetriever
import torch
import pinecone
from pinecone_text.sparse import SpladeEncoder
from langchain.embeddings import HuggingFaceEmbeddings
import os
import tiktoken

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Pinecone
api_key = "2c3790ff-1d6a-48be-b101-1301723b6252"
env = "us-east-1-aws"
pinecone.init(api_key=api_key, environment=env)
index = pinecone.Index("justiz")

# Loading models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splade = SpladeEncoder(device=device)
model_name = 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
embeddings = HuggingFaceEmbeddings(model_name=model_name)

retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=splade, index=index, top_k=40, alpha=0.3) #lower alpha - more sparse

question = "Beurteile den folgenden Fall: Adam arbeitet in der Autowerkstätte von Fred als angestellter Mechaniker. Ohne Wissen von Fred arbeitet Adam nachts des öfteren in einem call center für Milchprodukte. Manchmal arbeitet er dort die ganze Nacht und kommt nicht zum Schlafen. Als Fred davon erfärt, ist er empört. Kann er Adam diese Tätigkeit neben seinem Dienstverhältnis mit Fred verbieten?"
results = retriever.get_relevant_documents(question)

# Initialize tokenizer
helper = TokenizerHelper()
tokenizer = Tokenizer.from_file(helper.get_model_file("gpt4"))

# Create a prompt template
template = """
Du bist ein erfahrener Anwalt. Deine Aufgabe ist es die folgende Frage zu beantworten: {question}
Um die Frage zu beantworten hast du die folgenden Entscheidungen des Österreichischen Obersten Gerichtshofes zur Verfügung:
{sources}
Schreib eine ausführliche legale Analyse der Frage im Stil eines Rechtsgutachtens und gib für jede Aussage die entsprechende 'Quelle' an.
"""

token_count = len(list(tokenizer.tokenize(template))) + len(list(tokenizer.tokenize(question)))
print(token_count)
max_token_limit = 200
source_info = ""

for result in results:
    new_text = f"Inhalt: {result.page_content}\nQuelle: {result.metadata['source']}\n"
    new_tokens = list(tokenizer.tokenize(new_text))
    new_token_count = len(new_tokens)

    if token_count + new_token_count > max_token_limit:
        break

    source_info += new_text
    token_count += new_token_count
    print(token_count)

prompt_template = PromptTemplate.from_template(template)

gpt4prompt = prompt_template.format(question=question, sources=source_info)
print(gpt4prompt)
