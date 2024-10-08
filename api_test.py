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
gpt35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=5, openai_api_key=openai_api_key, max_retries=20)
tokenizer = tiktoken.encoding_for_model("gpt-4")


#init of prompt templates and system message
analysis_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt.")
analysis_template_string = """
Deine Aufgabe ist es die folgende Frage zu beantworten: 
"{question}"
Um die Frage zu beantworten hast du die folgenden Entscheidungen des Österreichischen Obersten Gerichtshofes zur Verfügung:
"{sources}"
Schreibe ein ausführliches Rechtsgutachten. Zuerst klärst du welche Rechtsfrage sich stellt.
Dann erörterst Du die Rechtsfrage abstrakt und beschreibst dabei jewils im Zuge der Erörterung einzelner Fragen auch die Fälle die vom Obersten Gerichtshof bereits entschieden wurden und gib dazu die Fallzahl an.
Vermeide aber eine bloße Auflistung der Fälle.
Danach wendest Du die so beschriebene Rechtslage auf den abgefragten Fall an.
Schließlich gib an, wie die Frage zu lösen ist. Falls die Lösung nicht eindeutig ist gib an, wie die wahrscheinlichere Lösung lautet. Gib auch an, welche zusätzlichen Sachverhaltselemente hilfreich wären.
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

ranking_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt. Deine Antwort besteht immer nur aus einer Zahl von 1 bis 10")
ranking_template_string = """
Deine Aufgabe ist es zu bewerten wie relevant ein Abschnitt einer Gerichtsentscheidung ist um die folgende Frage zu beantworten: 
"{question}"
Der Abschnitt lautet:
"{case}"
Skaliere die Relevanz auf einer Skala von 1 bis 10 und antworte mit dieser Zahl
"""
ranking_template = PromptTemplate.from_template(ranking_template_string)

ranking_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt. Deine Antwort besteht immer nur aus einer Zahl von 1 bis 10")
ranking_template_string = """
Deine Aufgabe ist es zu bewerten wie relevant ein Abschnitt einer Gerichtsentscheidung ist um die folgende Frage zu beantworten: 
"{question}"
Der Abschnitt lautet:
"{case}"
Skaliere die Relevanz auf einer Skala von 1 bis 10 und antworte mit dieser Zahl
"""
ranking_template = PromptTemplate.from_template(ranking_template_string)

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
    print(f"Using {nr_sources} chunks for analysis")
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
    retriever = PineconeHybridSearchRetriever(embeddings=dense_encoder, sparse_encoder=sparse_encoder, index=index, top_k=100, alpha=0.3) #lower alpha - more sparse
    return retriever

async def async_prune(case, question):
    pruning_userprompt = pruning_template.format(case=case.page_content, question=question)
    pruning_user_message = HumanMessage(content=pruning_userprompt)
    relevance = await gpt35._agenerate([pruning_system_message, pruning_user_message])
    print(case.page_content, "\n", relevance.generations[0].text)
    return (case, relevance.generations[0].text)

async def prune_concurrently(cases, question):
    tasks = [async_prune(case, question) for case in cases]
    results = await asyncio.gather(*tasks)
    #print(results)
    return [case for case, relevance in results if relevance == 'Ja.']

def prune_cases(results, question):
    pruned_results = asyncio.run(prune_concurrently(results, question))
    return pruned_results

async def async_rank(case, question):
    ranking_userprompt = ranking_template.format(case=case.page_content, question=question)
    ranking_user_message = HumanMessage(content=ranking_userprompt)
    relevance = await gpt35._agenerate([ranking_system_message, ranking_user_message])
    print(case.page_content, "\n", relevance.generations[0].text)
    return (case, float(relevance.generations[0].text))  # Ensure relevance is a number

async def rank_concurrently(cases, question):
    tasks = [async_rank(case, question) for case in cases]
    results = await asyncio.gather(*tasks)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)  # Sort by relevance
    return [case for case, _ in sorted_results]  # Only return the case objects

def rank_cases(results, question):
    ranked_results = asyncio.run(rank_concurrently(results, question))
    return ranked_results


def main():
    retriever = get_retriever()
    question = "Alfred kommt verspätet zur Arbeit. Er hat keine Entschuldigung und es ist ihm schon zum zweiten Mal passiert. Was sind die Konsequenzen?"
    #pinecone_prompt = 
    results = retriever.get_relevant_documents(question)

    print(f"{len(results)} chunks found in database")

    results = rank_cases(results=results, question=question)

    print(f"{len(results)} chunks left after pruning")

    max_tokens = ((gpt4_maxtokens - response_maxtokens) - 20) - (len(list(tokenizer.encode(analysis_template_string))) + len(list(tokenizer.encode(question))))
    sources = fill_tokens(results=results, max_tokens=max_tokens)

    analysis_userprompt = analysis_template.format(question=question, sources=sources)
    print(analysis_userprompt)
    user_message = HumanMessage(content=analysis_userprompt)
    response = gpt4([analysis_system_message, user_message])

if __name__ == "__main__":
    main()