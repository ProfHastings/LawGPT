import os
import time
import re
import requests
import subprocess
import chardet
import shutil
import concurrent.futures
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from urllib.parse import urlparse
from shutil import move


# Constants
MAX_DOWNLOAD_RETRIES = 500
MAX_PAGES = 2000
START_URL = 'https://www.ris.bka.gv.at/Ergebnis.wxe?Abfrage=Justiz&Fachgebiet=&Gericht=&Rechtssatznummer=&Rechtssatz=&Fundstelle=&Spruch=&Rechtsgebiet=Undefined&AenderungenSeit=Undefined&JustizEntscheidungsart=&SucheNachRechtssatz=False&SucheNachText=True&GZ=&VonDatum=&BisDatum=24.05.2023&Norm=&ImRisSeitVonDatum=&ImRisSeitBisDatum=&ImRisSeit=Undefined&ResultPageSize=100&Suchworte=&Position=1&Sort=1%7cAsc'
WORKING_DIRECTORY = 'working_dir'
DATABASE_DIRECTORY = 'test_database'
counter = 0

# Setup
if not os.path.exists(WORKING_DIRECTORY):
    os.mkdir(WORKING_DIRECTORY)
if not os.path.exists(DATABASE_DIRECTORY):
    os.mkdir(DATABASE_DIRECTORY)

# Selenium Chrome Driver Configuration
options = Options()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

def download_files(soup, base_url):
    """Download all rtf files from a single page"""
    links = [base_url + link.get('href') for link in soup.find_all('a') if link.get('href', '').endswith('.rtf')]
    for file_url in links:
        for _ in range(MAX_DOWNLOAD_RETRIES):
            try:
                time.sleep(0.1)
                file_response = requests.get(file_url)
                file_response.raise_for_status()

                # Sanitize filename
                filename = os.path.basename(urlparse(file_url).path)
                filename = re.sub('[^\w\-_\. ]', '_', filename)

                # Save RTF file
                rtf_file_path = os.path.join(WORKING_DIRECTORY, filename)
                with open(rtf_file_path, 'wb') as f:
                    f.write(file_response.content)

                break
            except requests.exceptions.RequestException:
                print(f"Download failed for '{file_url}', retrying.")

def convert_rtf_to_txt(directory):
    """Converts all rtf files in the given directory to txt files using LibreOffice."""
    # Define command
    command = "soffice --headless --convert-to txt:Text --outdir {} {}/*.rtf".format(directory, directory)

    # Execute command
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)

def clean_and_reencode_files(directory):
    """Cleans up the file content and re-encodes to UTF-8."""
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            full_path = os.path.join(directory, filename)
            
            # Detect the file encoding
            with open(full_path, 'rb') as f:
                result = chardet.detect(f.read())

            # Read the file using the detected encoding
            with open(full_path, 'r', encoding=result['encoding'], errors='ignore') as f:
                contents = f.readlines()

            # Clean up the content and re-encode to UTF-8
            cleaned_contents = [line.strip() for line in contents if line.strip()]
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(cleaned_contents))


def delete_residual_files(directory):
    """Deletes all non-txt files in the given directory."""
    for filename in os.listdir(directory):
        if not filename.endswith('.txt'):
            os.remove(os.path.join(directory, filename))

def move_files_to_database(src_directory, dst_directory):
    """Moves all txt files from source directory to destination directory."""
    global counter
    for filename in os.listdir(src_directory):
        if filename.endswith('.txt'):
            counter += 1
            print (counter)
            shutil.move(os.path.join(src_directory, filename), os.path.join(dst_directory, filename))
def main():
    # Visit the starting page
    driver.get(START_URL)
    # Main process: Scrape pages and download, convert, clean files
    for _ in range(MAX_PAGES):
        time.sleep(2)  # Allow page to load
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        base_url = 'https://www.ris.bka.gv.at'
        print("downloading...")
        download_files(soup, base_url)
        print("converting...")
        convert_rtf_to_txt(WORKING_DIRECTORY)
        print("cleaning...")
        clean_and_reencode_files(WORKING_DIRECTORY)
        print("deleting...")
        delete_residual_files(WORKING_DIRECTORY)
        print("moving...")
        move_files_to_database(WORKING_DIRECTORY, DATABASE_DIRECTORY)

        # Go to next page
        try:
            wait = WebDriverWait(driver, 10)
            next_button = wait.until(EC.element_to_be_clickable((By.ID, 'PagingTopControl_NextPageLink')))
            next_button.click()
            time.sleep(2)
        except Exception:
            print("No more pages to scrape. Exiting...")
            break

    driver.quit()

if __name__ == "__main__":
    main()