import requests
from bs4 import BeautifulSoup
import os
import time
from striprtf.striprtf import rtf_to_text

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1'})

url = 'https://www.example.com'
response = session.get(url)

soup = BeautifulSoup(response.text, 'html.parser')
links = soup.find_all('a')

for link in links:
    file_url
    file_url = link.get('href')
    if file_url.endswith('.rtf'):  # Check if the href is a link to a .rtf file
        time.sleep(5)  # sleep for 5 seconds between requests
        file_response = session.get(file_url)
        
        # Convert bytes to text using rtf_to_text function from striprtf
        text_content = rtf_to_text(file_response.content.decode('latin-1'))  # Assuming the .rtf file is encoded in 'latin-1'
        
        # Save the text content in a .txt file in the current directory
        with open(os.path.join('./', os.path.basename(file_url).replace('.rtf', '.txt')), 'w') as f:
            f.write(text_content)
