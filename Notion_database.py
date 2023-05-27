#keinen Blödsinn anstellen :)
#einmal einfang 

#Ich versuche Daten in eine Datenbank bei mir hinzuzufügen
import requests

Notion_token = "secret_F2Sj4AGEJLPgaIHO45HJyFMOynsGDxd846SFoSk32Od"
database_id = "9857222d269f43739679e1515935b978"

headers = {
    "Authorization": "Bearer " + Notion_token,
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}


def get_pages(num_pages=None):
    """
    If num_pages is None, get all pages, otherwise just the defined number.
    """
    url = f"https://api.notion.com/v1/databases/{database_id}/query"

    get_all = num_pages is None
    page_size = 100 if get_all else num_pages

    payload = {"page_size": page_size}
    response = requests.post(url, json=payload, headers=headers)

    data = response.json()

    # Comment this out to dump all data to a file
    # import json
    # with open('db.json', 'w', encoding='utf8') as f:
    #    json.dump(data, f, ensure_ascii=False, indent=4)

    results = data["results"]
    while data["has_more"] and get_all:
        payload = {"page_size": page_size, "start_cursor": data["next_cursor"]}
        url = f"https://api.notion.com/v1/databases/{database_id}/query"
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        results.extend(data["results"])
    return results


pages = get_pages()

for page in pages:
    page_id = page["id"]
    props = page["properties"]
    bezeichnung = props["Bezeichnung"]["title"][0]["text"]["content"]
    name = props["Text"]["rich_text"][0]["text"]["content"]
    datum = props["Datum"]["rich_text"][0]["text"]["content"]
    print(bezeichnung,name,datum)


def create_page(data: dict):
    create_url = "https://api.notion.com/v1/pages"

    payload = {"parent": {"database_id": database_id}, "properties": data}

    res = requests.post(create_url, headers=headers, json=payload)
    # print(res.status_code)
    return res

title = "Test Title"

file_path = r"C:\Users\canti\Documents\GitHub\LawGPT\requirements_scraper.txt"
with open(file_path, "r") as file:
    text = file.read()

datum = "Test Datum"
data = {
    "Bezeichnung": {"title": [{"text": {"content": title}}]},
    "Text": {"rich_text": [{"text": {"content": text}}]},
    "Datum": {"rich_text": [{"text": {"content": datum}}]}
}

create_page(data)