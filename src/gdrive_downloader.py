import gdown
import os


def download_model_from_google_drive(model_id, destination):
    if not os.path.exists(destination):
        url = f'https://drive.google.com/uc?id={model_id}'
        gdown.download(url, destination, quiet=False)
