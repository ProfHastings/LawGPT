from langchain import PromptTemplate
from langchain.retrievers import PineconeHybridSearchRetriever
import torch
import pinecone
from pinecone_text.sparse import SpladeEncoder
import os
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
#    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import asyncio
from langchain.embeddings import OpenAIEmbeddings
import gc

#init of global gpt-4 model, gpt-3.5-turbo model and OpenAI tokenizer
gpt4_maxtokens = 8192
response_maxtokens = 2048
callback_handler = [StreamingStdOutCallbackHandler()]
gpt4 = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=2048, streaming=True, callbacks=callback_handler)
gpt35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=5, max_retries=20)
gptdataquery = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=256)
tokenizer = tiktoken.encoding_for_model("gpt-4")


#template for final analysis prompt
analysis_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt.")
analysis_template_string = """
Deine Aufgabe ist es die folgende Frage zu beantworten: 
"{question}"
Um die Frage zu beantworten hast du die folgenden Entscheidungen des Österreichischen Obersten Gerichtshofes zur Verfügung:
"{sources}"
Du bist Rechtsanwältsanwärter in einer Anwaltskanzlei. Schreibe einen sehr ausführlichen und detaillierten ersten Entwurf für ein Rechtsgutachten. Zuerst klärst du welche Rechtsfrage sich stellt.
Dann erörterst Du die Rechtsfrage abstrakt und nimmst dabei jeweils im Zuge der Erörterung einzelner Fragen auch auf Fälle Bezug, die vom Obersten Gerichtshof bereits entschieden wurden und gib dazu die Fallzahl an.
Vermeide aber eine bloße Auflistung der Fälle.
Danach wendest Du die so beschriebene Rechtslage auf den Fall an.
Schließlich gib an, wie die Frage deines Erachtenszu lösen ist. Gib immer auch an, wenn du dich in deinen Ausführungen unsicher fühlst.Falls die Lösung nicht eindeutig ist gib an, wie die wahrscheinlichere Lösung lautet. Gib auch an, welche zusätzlichen Sachverhaltselemente hilfreich wären.
Zum Schluß liste die fünf wichtigsten Entscheidungen und die fünf wichtigsten Literaturzitate auf, die du in den Entscheidungen findest.
"""
analysis_template = PromptTemplate.from_template(analysis_template_string)

#template for pruning prompt
pruning_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt. Deine Antwort besteht immer nur aus einem Wort")
pruning_template_string = """
Deine Aufgabe ist es zu evaluieren ob ein Abschnitt einer Gerichtsentscheidung relevant sein könnte um die folgende Frage zu beantworten: 
"{question}"
Der Abschnitt lautet:
"{case}"
Falls du dir sicher bist, dass der Abschnitt die Rechtsfrage nicht betrifft, antworte mit Nein. Ansonsten antworte mit Ja.
"""
pruning_template = PromptTemplate.from_template(pruning_template_string)

#template for ranking prompt
ranking_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt. Deine Antwort besteht immer nur aus einer Zahl von 1 bis 10")
ranking_template_string = """
Deine Aufgabe ist es zu bewerten wie relevant ein Abschnitt einer Gerichtsentscheidung ist um die folgende Frage zu beantworten: 
"{question}"
Der Abschnitt lautet:
"{case}"
Skaliere die Relevanz auf einer Skala von 1 bis 10 und antworte mit dieser Zahl
"""
ranking_template = PromptTemplate.from_template(ranking_template_string)

#template for database query prompt
dataquery_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt. Du antwortest nur genau mit dem was von dir gefragt ist und ausführlich.")
dataquery_template_string = """
Ein Klient kommt zu dir mit der folgenden Frage.
"{question}"
Schreibe eine Liste mit den wichtigsten rechtlichen Fragen die sich zu dieser Situation stellen. Verwende die genaue juristische Terminologie.
"""
#dataquery_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt. Du schreibst nur die Antwort auf Fragen.")
#dataquery_template_string = """
#Du hast die folgende informell formulierte Frage:"
#"{question}" 
#Formuliere die Frage ausführlich um, so wie sie in einer gerichtlichen Entscheidung stehen würde.
#"""
#dataquery_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt. Du antwortest nur genau mit dem was von dir gefragt ist und ausführlich.")
#dataquery_template_string = """
#Ein Klient kommt zu dir mit der folgenden Frage.
#"{question}"
#Identifiziere die rechtlichen Parteien in diesem Fall und nutze dies um eine Liste mit den drei wichtigsten Fragen die sich zu dieser Situation zu schreiben. Nenne die Parteien innerhalt der Liste nur mit ihren rechtlichen Rollen ohne persönliche Namen und schreibe nur die Elemente der Liste ohne Erläuterung. Jedes Element der Liste sollte formuliert sein um die Parteien und Rechtsinteraktion zu referenzieren. Jeder der Punkte soll verständlich sein ohne die Originalfrage gelesen zu haben und die Rechtslage enthalten. Verwende die genaue juristische Terminologie.
#"""

dataquery_template = PromptTemplate.from_template(dataquery_template_string)

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
    api_key = "953b2be8-0621-42a1-99db-8480079a9e23"
    env = "eu-west4-gcp"
    pinecone.init(api_key=api_key, environment=env)
    return pinecone.Index("justiz-openai")

#loads dense and sparse encoder models and returns retriever to send requests to the database
def get_retriever():
    index = get_index()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense_encoder = OpenAIEmbeddings(model="text-embedding-ada-002")
    sparse_encoder = SpladeEncoder(device=device)
    retriever = PineconeHybridSearchRetriever(embeddings=dense_encoder, sparse_encoder=sparse_encoder, index=index, top_k=75, alpha=0.99899) #lower alpha - more sparse
    return retriever

#(next three functions) uses async api calls to prune cases based on relevance
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

#(next three functions) uses async api calls to rank chunks from 1-10 based on relevance
async def async_rank(case, question, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            ranking_userprompt = ranking_template.format(case=case.page_content, question=question)
            ranking_user_message = HumanMessage(content=ranking_userprompt)
            relevance = await gpt35._agenerate([ranking_system_message, ranking_user_message])
            relevance_score = float(relevance.generations[0].text)
            print(case.page_content, "\n", relevance_score, "\n")
            return (case, relevance_score)
        except ValueError:
            print(f"Attempt {attempt + 1} failed, did not return ranking number")
    print(f"All {max_attempts} attempts failed. Returning default relevance score of 1.")
    return (case, 1)

async def rank_concurrently(cases, question):
    tasks = [async_rank(case, question) for case in cases]
    results = await asyncio.gather(*tasks)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    sum_of_relevance = sum(relevance for _, relevance in results)
    return [case for case, _ in sorted_results], sum_of_relevance

def rank_cases(results, question):
    ranked_results, sum_of_relevance = asyncio.run(rank_concurrently(results, question))
    print(f"Average relevance score: {sum_of_relevance/len(ranked_results)}")
    return ranked_results

#rephrases query as optimized prompt for searching vectorstorage
def get_dataquery(question):
    dataquery_userprompt = dataquery_template.format(question=question)
    dataquery_user_message = HumanMessage(content=dataquery_userprompt)
    dataquery = gptdataquery([dataquery_system_message, dataquery_user_message])
    print(f"Looking in database for: {dataquery.content}")
    return dataquery.content

#attempt to force garbace collection. seems unsuccessful
def smart_retriever(question):
    index = get_index()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense_encoder = OpenAIEmbeddings(model="text-embedding-ada-002")
    sparse_encoder = SpladeEncoder(device=device)
    retriever = PineconeHybridSearchRetriever(embeddings=dense_encoder, sparse_encoder=sparse_encoder, index=index, top_k=75, alpha=0.99899) #lower alpha - more sparse
    
    dataquery = get_dataquery(question)
    print(f"Looking in database for: {dataquery}")  

    results = retriever.get_relevant_documents(dataquery)
    del (dense_encoder, sparse_encoder, index, device, retriever, dataquery)
    torch.cuda.empty_cache()
    gc.collect()
    return results

def main(question):
    retriever = get_retriever()
    dataquery = get_dataquery(question)
    #dataquery = "Verletzt der Arbeitgeber schuldhaft seine Fürsorgepflicht und entsteht dem Arbeitnehmer ein Schaden, so trifft den Arbeitgeber eine Schadenersatzpflicht (vgl Pfeil in Schwimann, ABGB³ V § 1157 Rz 32; Marhold in Marhold/Burgstaller/Preyer, AngG § 18 Rz 120 ua). Der Kläger macht Gesundheitsschäden und damit zusammenhängenden Verdienstentgang und sonstige Kosten geltend, die auf die Verletzung der Abhilfeverpflichtung der Beklagten zurückzuführen sein sollen. Diese Schadenersatzansprüche unterliegen den allgemeinen Voraussetzungen des Schadenersatzrechts (vgl Mosler in ZellKomm² AngG § 18 Rz 132 ua), insbesondere auch in Bezug auf das Vorliegen eines Schadens und dessen Verursachung durch den Schädiger. Für beides trägt der Geschädigte die Beweislast. Der Kläger hat dazu auch entsprechende Behauptungen in erster Instanz aufgestellt, die von der Beklagten bestritten wurden. Dafür, dass beim Kläger eine psychische Erkrankung eingetreten ist, scheinen vom Kläger vorgelegte ärztliche Befunde zu sprechen. Konkrete Tatsachenfeststellungen des Erstgerichts oder Außerstreitstellungen der Parteien dazu fehlen aber bisher. Da die vom Kläger behauptete psychische Erkrankung bisher nicht festgestellt wurde, wurden auch keine Feststellungen getroffen, wodurch diese Erkrankung nun tatsächlich verursacht wurde. Der Kläger steht auf dem Standpunkt, dass seine psychische Erkrankung auf die von der Beklagten nicht unterbundenen Beschimpfungen und Schikanen zurückzuführen sei. Dies wurde von der Beklagten bestritten. Die Frage der Verursachung der vom Kläger verursachten Schäden harrt daher einer Klärung im zweiten Rechtsgang. Dabei ist auf den Zeitraum der Verletzung der Fürsorgepflicht ab 7. 11. 2008 abzustellen. "
    results = retriever.get_relevant_documents(dataquery)
    #results = smart_retriever(question)
    #gc.collect()
    #for result in results:
    #    print (result.page_content, "\n", "\n")
    #return
    #print(f"{len(results)} chunks found in database")

    results = rank_cases(results=results, question=question)
    
    #print(results)
    
    max_tokens = ((gpt4_maxtokens - response_maxtokens) - 20) - (len(list(tokenizer.encode(analysis_template_string))) + len(list(tokenizer.encode(question))))
    sources = fill_tokens(results=results, max_tokens=max_tokens)
    analysis_userprompt = analysis_template.format(question=question, sources=sources)
    print(analysis_userprompt)
    user_message = HumanMessage(content=analysis_userprompt)
    response = gpt4([analysis_system_message, user_message])
    return response.content

if __name__ == "__main__":
    main("Alfred arbeitet in einer Fabrik und schläft wo während er am Fließband arbeitet. Es entsteht ein erheblicher Schaden. Kann er zu Schadenersatz verurteilt werden?")