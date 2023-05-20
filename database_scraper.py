from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import os
import time
import re
from striprtf.striprtf import rtf_to_text
from requests.exceptions import RequestException
from urllib.parse import urlparse
import requests
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# Maximum number of times to retry a download
MAX_DOWNLOAD_RETRIES = 100

# Maximum number of pages to scrape
MAX_PAGES = 1500  # Adjust as necessary

# Set up Selenium
options = Options()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

url = 'https://www.ris.bka.gv.at/Ergebnis.wxe?Abfrage=Justiz&Fachgebiet=&Gericht=&Rechtssatznummer=&Rechtssatz=&Fundstelle=&Spruch=&Rechtsgebiet=Undefined&AenderungenSeit=Undefined&JustizEntscheidungsart=&SucheNachRechtssatz=False&SucheNachText=True&GZ=&VonDatum=&BisDatum=19.05.2023&Norm=&ImRisSeitVonDatum=&ImRisSeitBisDatum=&ImRisSeit=Undefined&ResultPageSize=100&Suchworte=&Position=1&Sort=1%7cAsc'
driver.get(url)

save_directory = 'database'
counter = 0

# Iterate over each page
for _ in range(MAX_PAGES):
    # Let the page load
    time.sleep(2)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    base_url = 'https://www.ris.bka.gv.at'
    links = [base_url + link.get('href') for link in soup.find_all('a') if link.get('href', '').endswith('.rtf')]

    for file_url in links:
        counter += 1
        print(counter)
        for i in range(MAX_DOWNLOAD_RETRIES):
            try:
                time.sleep(0.1)
                file_response = requests.get(file_url)
                # Check status code of HTTP response
                file_response.raise_for_status()

                # Convert bytes to text using rtf_to_text function from striprtf
                text_content = rtf_to_text(file_response.content.decode('utf-8'))  # Assuming the .rtf file is encoded in 'utf-8'
                # Remove leading whitespace and empty lines
                text_content = '\n'.join(line.lstrip() for line in text_content.splitlines() if line.strip())
                # Filename sanitization
                filename = os.path.basename(urlparse(file_url).path)
                lines = text_content.splitlines()
                if len(lines) >= 6 and lines[4] == "Geschäftszahl":
                    filename = lines[5] + '.txt'
                filename = re.sub('[^\w\-_\. ]', '_', filename)  # Replace all non-alphanumeric or underscore or hyphen or period or space characters with underscore
                filename = filename.replace('.rtf', '.txt')

                # Save the text content in a .txt file in the save directory
                with open(os.path.join(save_directory, filename), 'w', encoding='utf-8') as f:
                    f.write(text_content)
                
                # Exit the retry loop if the download was successful
                break
            except RequestException:
                print(f"Download failed for '{file_url}', retrying ({i+1}/{MAX_DOWNLOAD_RETRIES})")

    # Check if there's a next page and if so, navigate to it
    try:
        # Wait until the next button is clickable
        print("clicking button")
        wait = WebDriverWait(driver, 10)
        next_button = wait.until(EC.element_to_be_clickable((By.ID, 'PagingTopControl_NextPageLink')))
        print(next_button)
        next_button.click()
        time.sleep(2)
    except Exception as e:
        print("No more pages to scrape. Exiting...")
        break

driver.quit()
