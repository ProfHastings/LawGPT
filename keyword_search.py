import os
import re
import codecs
import chardet

def find_files_with_keyword(dir_path, keyword):
    counter = 0
    files_with_keyword = []

    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                counter += 1
                if(counter % 1000 == 0):
                    print(counter)
                try:
                    rawdata = open(file_path, 'rb').read()
                    result = chardet.detect(rawdata)
                    charenc = result['encoding']

                    if charenc != 'utf-8':
                        print(f"File {file_path} is not encoded with utf-8, detected encoding is {charenc}")

                    with codecs.open(file_path, 'r', encoding=charenc) as f:
                        content = f.read()
                        if re.search(keyword, content):
                            files_with_keyword.append(file_path)
                except Exception as e:
                    print(f"Could not read {file_path} due to {e}")

    return files_with_keyword

dir_path = 'database - Copy'  # replace with your directory path
keyword = '§228'  # replace with your keyword

files_with_keyword = find_files_with_keyword(dir_path, keyword)

for file in files_with_keyword:
    print(file)
