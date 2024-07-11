import torch
from PIL import Image
import requests
from io import BytesIO
import logging
from RealESRGAN import RealESRGAN
from ilia_real_esrgan import IliaRealESRGAN
from gdrive_downloader import download_model_from_google_drive

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_image_with_model(model_type, image_url):
    response = requests.get(image_url)
    input_image = Image.open(BytesIO(response.content))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'model_real_esrgan':
        model = RealESRGAN(device, scale=2)
        model.load_weights('weights/RealESRGAN_x2.pth', download=True)
        sr_image = model.predict(input_image)
        sr_image.save('output.png')
    elif model_type == 'model_ilia_real_esrgan':
        model_path = 'ilia_real_esrgan.pth'
        download_model_from_google_drive('1wv6433R7yAgDblofyo-vTSi3ZmQ7oW_J', model_path)
        model = IliaRealESRGAN(device, model_path)
        sr_image = model.predict(input_image)
        sr_image.save('output.png')
    else:
        input_image.save('output.png')
    return 'output.png'
