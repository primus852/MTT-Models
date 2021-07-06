from typing import Any

import requests


class Downloader:

    @staticmethod
    def download_file_from_google_drive(file_id: str, full_path: str) -> None:
        url = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(url, params={'id': file_id}, stream=True)
        token = Downloader.get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        Downloader.save_response_content(response, full_path)

    @staticmethod
    def get_confirm_token(response) -> Any:
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    @staticmethod
    def save_response_content(response, destination) -> None:
        chunk_size = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
