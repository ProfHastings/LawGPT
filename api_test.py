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
import asyncio

#init of global gpt-4 model, gpt-3.5-turbo model and OpenAI tokenizer
gpt4_maxtokens = 8192
response_maxtokens = 2048
openai_api_key = "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"
callback_handler = [StreamingStdOutCallbackHandler()]
gpt4 = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=2048, streaming=True, callbacks=callback_handler, openai_api_key=openai_api_key)
gpt35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=5, openai_api_key=openai_api_key)

tokenizer = tiktoken.encoding_for_model("gpt-4")


#init of prompt templates and system message
analysis_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt.")
analysis_template_string = """
Deine Aufgabe ist es die folgende Frage zu beantworten: 
"{question}"
Um die Frage zu beantworten hast du die folgenden Entscheidungen des Österreichischen Obersten Gerichtshofes zur Verfügung:
"{sources}"
Schreib eine ausführliche rechtliche Analyse der Frage im Stil eines Rechtsgutachtens und gib für jede Aussage die entsprechende Quelle in Klammer an.
Falls vorhanden, gehe in deinen Ausführungen auf vergleichbare Fälle ein, die du in den Entscheidungen des Obersten Gerichtshofs findest. 
Schließlich gib an, wie die Frage zu lösen ist. Falls die Lösung nicht eindeutig ist gib an, wie die wahrscheinlichere Lösung lautet. Gib auch an, welche zusätzlichen SAchverhaltselemente hilfreich wären.
"""
analysis_template = PromptTemplate.from_template(analysis_template_string)

pruning_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt. Deine Antwort besteht immer nur aus einem Wort")
pruning_template_string = """
Deine Aufgabe ist es zu evaluieren ob ein Abschnitt einer Gerichtsentscheidung relevant sein könnte um die folgende Frage zu beantworten: 
"{question}"
Der Abschnitt lautet:
"{case}"
Falls du dir sicher bist, dass der Abschnitt die Rechtsfrage nicht betrifft, antworte mit Nein. Ansonsten antworte mit Ja.
"""
pruning_template = PromptTemplate.from_template(pruning_template_string)

#takes Vector Database results, returns highest number of results with sources as string that fit in max_tokens OpenAI
def fill_tokens(results, max_tokens):
    sources = ""
    nr_sources = 0
    token_count = 0
    
    for result in results:
        new_text = f"Inhalt: {result.page_content}\nQuelle: {result.metadata['short_source']}\n"
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

#loads dense and sparse encoder models and returns retriever to send requests to the database
def get_retriever():
    index = get_index()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense_model_name = 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
    dense_encoder = HuggingFaceEmbeddings(model_name=dense_model_name)
    sparse_encoder = SpladeEncoder(device=device)
    retriever = PineconeHybridSearchRetriever(embeddings=dense_encoder, sparse_encoder=sparse_encoder, index=index, top_k=150, alpha=0.3) #lower alpha - more sparse
    return retriever

async def async_generate(case, question):
    pruning_userprompt = pruning_template.format(case=case.page_content, question=question)
    #print(pruning_userprompt)
    pruning_user_message = HumanMessage(content=pruning_userprompt)
    relevance = gpt35([pruning_system_message, pruning_user_message])
    print(case.page_content)
    print(relevance.content)
    return (case, relevance.content)

async def generate_concurrently(cases, question):
    tasks = [async_generate(case, question) for case in cases]
    results = await asyncio.gather(*tasks)
    #print(results)
    return [case for case, relevance in results if relevance == 'Ja.']

def prune_cases(results, question):
    pruned_results = asyncio.run(generate_concurrently(results, question))
    return pruned_results


def main():
    retriever = get_retriever()
    question = "Alfred kommt verspätet zur Arbeit. Er hat keine Entschuldigung und es ist ihm schon zum zweiten Mal passiert. Was sind die Konsequenzen?"
    results = retriever.get_relevant_documents(question)

    print(len(results))

    #implement case pruning
    results = prune_cases(results=results, question=question)

    print(len(results))
    #print(results)
    
    
    
    max_tokens = ((gpt4_maxtokens - response_maxtokens) - 20) - (len(list(tokenizer.encode(analysis_template_string))) + len(list(tokenizer.encode(question))))
    sources = fill_tokens(results=results, max_tokens=max_tokens)

    analysis_userprompt = analysis_template.format(question=question, sources=sources)
    print(analysis_userprompt)
    user_message = HumanMessage(content=analysis_userprompt)
    response = gpt4([analysis_system_message, user_message])

if __name__ == "__main__":
    main()