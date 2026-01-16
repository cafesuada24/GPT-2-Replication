import os
import zipfile
from pathlib import Path
import requests


def download_and_unzip_spam_data(
    url: str,
    zip_path: str | Path,
    extracted_path: str | Path,
    data_file_path: str | Path,
):
    data_file_path = Path(data_file_path)
    if data_file_path.exists():
        print(f'{data_file_path} already exists. Skipping download and extraction.')
        return

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(zip_path, 'wb') as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / 'SMSSpamCollection'
    os.rename(original_file_path, data_file_path)
    print(f'Data downloaded and saved as {data_file_path}')


if __name__ == '__main__':
    url = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
    zip_path = 'sms_spam_collection.zip'
    extracted_path = 'data/sms_spam_collection'
    data_file_path = Path(extracted_path) / 'SMSSpamCollection.tsv'

    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
