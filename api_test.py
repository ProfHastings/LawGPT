from langchain import PromptTemplate
from langchain.retrievers import PineconeHybridSearchRetriever
import torch
import pinecone
from pinecone_text.sparse import SpladeEncoder
from langchain.embeddings import HuggingFaceEmbeddings
import os
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


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
callback_handler = [StreamingStdOutCallbackHandler()]
gpt4 = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=2048, streaming=True, callbacks=callback_handler)

retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=splade, index=index, top_k=50, alpha=0.3) #lower alpha - more sparse

question = "Fred arbeitet für Max in einem Tonstudio. Fred macht eine tolle Erfindung, die es ihm ermöglicht auf elektronischem Weg Pfurtzgeräusche zu erzeugen. Zu einem geringen Teil hat er an dieser Erfindung während seiner Arbeitszeit gearbeitet. Wem gehört die Erfindung?"
results = retriever.get_relevant_documents(question)

tokenizer = tiktoken.encoding_for_model("gpt-4") 

template = """
Deine Aufgabe ist es die folgende Frage zu beantworten: "{question}"
Um die Frage zu beantworten hast du die folgenden Entscheidungen des Österreichischen Obersten Gerichtshofes zur Verfügung:
"{sources}"
Schreib eine ausführliche legale Analyse der Frage im Stil eines Rechtsgutachtens und gib für jede Aussage die entsprechende 'Quelle' an.
Beschreibe die Rechtsfrage abstrakt und ergänze deine Ausführungen mit praktischen Besipielen, die du in den Entscheidungen des Obersten Gerichtshofs findest. 
Vergleiche diese Besipiele auch mit dem Fall der der Frage zugrunde liegt.
"""

token_count = len(list(tokenizer.encode(template))) + len(list(tokenizer.encode(question)))
max_token_limit = 6100
#6144
#lower bc system
source_info = ""
nrsources = 0 
for result in results:
    new_text = f"Inhalt: {result.page_content}\nQuelle: {result.metadata['source']}\n"
    new_tokens = list(tokenizer.encode(new_text))
    new_token_count = len(new_tokens)

    if token_count + new_token_count > max_token_limit:
        break

    source_info += new_text
    token_count += new_token_count
    nrsources += 1

print(f"Used {nrsources} sources")
prompt_template = PromptTemplate.from_template(template)

gpt4userprompt = prompt_template.format(question=question, sources=source_info)
#print(gpt4userprompt)

system_message = SystemMessage(content="Du bist ein erfahrener Anwalt mit dem Spezialgebiet österreichisches Recht.")

user_message = HumanMessage(content=gpt4userprompt)

for response in gpt4([system_message, user_message]):
    print(response.content)

