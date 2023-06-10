from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import os
import time
import re
from requests.exceptions import RequestException
from urllib.parse import urlparse
import requests
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import subprocess
import chardet

# Maximum number of times to retry a download
MAX_DOWNLOAD_RETRIES = 100

# Maximum number of pages to scrape
MAX_PAGES = 1500  # Adjust as necessary

# Set up Selenium
options = Options()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

url = 'https://www.ris.bka.gv.at/Ergebnis.wxe?Abfrage=Justiz&Fachgebiet=&Gericht=&Rechtssatznummer=&Rechtssatz=&Fundstelle=&Spruch=&Rechtsgebiet=Undefined&AenderungenSeit=Undefined&JustizEntscheidungsart=&SucheNachRechtssatz=False&SucheNachText=True&GZ=&VonDatum=&BisDatum=05.06.2023&Norm=&ImRisSeitVonDatum=&ImRisSeitBisDatum=&ImRisSeit=Undefined&ResultPageSize=100&Suchworte=AngG&Position=1&SkipToDocumentPage=true'
driver.get(url)


counter = 0

# Create an empty list to store the URLs of the files that failed to decode
failed_urls = []

working_directory = os.getcwd()
save_directory = 'AngG'
absolute_directory = os.path.join(working_directory, save_directory)

# Function to convert rtf to txt using LibreOffice
import chardet

def convert_to_txt(filename):
    try:
        #print("Converting...")
        full_path = os.path.join(save_directory, filename)
        subprocess.check_call(['soffice', '--headless', '--convert-to', 'txt:Text', '--outdir', save_directory, full_path])
        txt_file = os.path.splitext(os.path.basename(filename))[0] + '.txt'

        #print(f"Original RTF File: {filename}")
        #print(f"Converted TXT File: {txt_file}")

        # Detect the encoding of the file
        with open(os.path.join(save_directory, txt_file), 'rb') as f:
            result = chardet.detect(f.read())

        # Open the file in the original encoding and read it, replacing any problematic characters with "?"
        try:
            with open(os.path.join(save_directory, txt_file), 'r', encoding=result['encoding'], errors='replace') as f:
                contents = f.read()
        except UnicodeDecodeError as e:
            print(f"Unicode decoding error for '{filename}', error message: {e}")
            failed_urls.append(filename)  # Add the filename to the list
            return None

        # Write the contents back into the file with UTF-8 encoding
        with open(os.path.join(save_directory, txt_file), 'w', encoding='utf-8') as f:
            f.write(contents)

        # Clean up the file content
        clean_file_content(os.path.join(save_directory, txt_file))

        # Remove the original RTF file
        os.remove(full_path)

        return contents

    except subprocess.CalledProcessError as e:
        print(f"RTF to TXT conversion failed for '{filename}', error message: {e}")
        failed_urls.append(filename)  # Add the filename to the list
        return None
    except FileNotFoundError as e:
        print(f"File '{filename}' not found in '{save_directory}', error message: {e}")
        return None




def clean_file_content(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        cleaned_line = line.strip()  # Remove leading and trailing spaces
        if cleaned_line:  # Skip empty lines
            cleaned_lines.append(cleaned_line)

    # Write cleaned up content back into the file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines))


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

                # Filename sanitization
                filename = os.path.basename(urlparse(file_url).path)
                filename = re.sub('[^\w\-_\. ]', '_', filename)  # Replace all non-alphanumeric or underscore or hyphen or period or space characters with underscore

                # Save the RTF file in the save directory
                rtf_file_path = os.path.join(save_directory, filename)
                with open(rtf_file_path, 'wb') as f:
                    f.write(file_response.content)
                #print("RTF FILE PATH:")
                #print(rtf_file_path)
                # Convert the saved RTF file to TXT
                convert_to_txt(filename)  # Pass filename

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
        next_button.click()
        time.sleep(2)
    except Exception as e:
        print("No more pages to scrape. Exiting...")
        print("\nConversion to text failed for the following URLs:")
        for url in failed_urls:
            print(url)
        break

driver.quit()
