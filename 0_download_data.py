import gdown
import zipfile
import os
from os.path import join
from os.path import isdir, isfile

output = './'

## download data from google drive
url = "https://drive.google.com/drive/folders/1Wc5ZZYDsAjMyakrx7Cy0a1QSD1vr0aGJ?usp=drive_link"
output_dir = join(output, 'data')
gdown.download_folder(url, output=output, quiet=True, use_cookies=False)

## unzip data folders
data_names = [join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.zip')]
for data_name in data_names:
    with zipfile.ZipFile(data_name, 'r') as zip_ref:
        zip_ref.extractall(output_dir)


