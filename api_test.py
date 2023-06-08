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

#init of global gpt4 model and OpenAI tokenizer
callback_handler = [StreamingStdOutCallbackHandler()]
gpt4 = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=2048, streaming=True, callbacks=callback_handler, openai_api_key="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")
tokenizer = tiktoken.encoding_for_model("gpt-4")

gpt4_maxtokens = 8192
response_maxtokens = 2048

#init of prompt templates and system message
system_message = SystemMessage(content="Du bist ein erfahrener Anwalt mit dem Spezialgebiet österreichisches Recht.")

analysis_template_string = """
Deine Aufgabe ist es die folgende Frage zu beantworten: "{question}"
Um die Frage zu beantworten hast du die folgenden Entscheidungen des Österreichischen Obersten Gerichtshofes zur Verfügung:
"{sources}"
Schreib eine ausführliche legale Analyse der Frage im Stil eines Rechtsgutachtens und gib für jede Aussage die entsprechende Quelle in Klammer an.
Beschreibe die Rechtsfrage abstrakt und ergänze deine Ausführungen mit praktischen Besipielen, die du in den Entscheidungen des Obersten Gerichtshofs findest. 
Vergleiche diese Beispiele auch mit dem Fall der der Frage zugrunde liegt.
"""
analysis_template = PromptTemplate.from_template(analysis_template_string)

#takes Vector Database results, returns highest number of results with sources as string that fit in max_tokens OpenAI
def fill_tokens(results, max_tokens):
    sources = ""
    nr_sources = 0
    token_count = 0
    
    for result in results:
        new_text = f"Inhalt: {result.page_content}\nQuelle: {result.metadata['source']}\n"
        new_tokens = list(tokenizer.encode(new_text))
        new_token_count = len(new_tokens)
        if token_count + new_token_count > max_tokens:
            break
        sources += new_text
        token_count += new_token_count
        nr_sources += 1
    print(f"Used {nr_sources} sources")
    return sources

#initializes Pinecone index to make database requests
def get_index():
    api_key = "2c3790ff-1d6a-48be-b101-1301723b6252"
    env = "us-east-1-aws"
    pinecone.init(api_key=api_key, environment=env)
    index = pinecone.Index("justiz")
    return index

#loads dense and sparse encoder models and 
def get_retriever():
    index = get_index()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense_model_name = 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
    dense_encoder = HuggingFaceEmbeddings(model_name=dense_model_name)
    sparse_encoder = SpladeEncoder(device=device)
    retriever = PineconeHybridSearchRetriever(embeddings=dense_encoder, sparse_encoder=sparse_encoder, index=index, top_k=50, alpha=0.3) #lower alpha - more sparse
    return retriever

def main():
    retriever = get_retriever()
    question = "Fred arbeitet für Max in einem Tonstudio. Fred macht eine tolle Erfindung, die es ihm ermöglicht auf elektronischem Weg Pfurtzgeräusche zu erzeugen. Zu einem geringen Teil hat er an dieser Erfindung während seiner Arbeitszeit gearbeitet. Wem gehört die Erfindung?"
    results = retriever.get_relevant_documents(question)
    max_tokens = ((gpt4_maxtokens - response_maxtokens) - 20) - (len(list(tokenizer.encode(analysis_template_string))) + len(list(tokenizer.encode(question))))
    sources = fill_tokens(results=results, max_tokens=max_tokens)

    analysis_userprompt = analysis_template.format(question=question, sources=sources)
    user_message = HumanMessage(content=analysis_userprompt)
    try:
        for response in gpt4([system_message, user_message]):
            print(response.content)
    except Exception as e:
        print()
if __name__ == "__main__":
    main()