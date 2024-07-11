import pytest  # noqa: F401
import torch
from PIL import Image

from src.ilia_real_esrgan import ResidualDenseBlock, ResidualInResidualDenseBlock, Generator, IliaRealESRGAN
from src.gdrive_downloader import download_model_from_google_drive


def test_residual_dense_block():
    block = ResidualDenseBlock()
    dummy_input = torch.randn(1, 64, 64, 64)
    output = block(dummy_input)
    assert dummy_input.shape == output.shape, "Output shape should match input shape due to residual connection"


def test_residual_in_residual_dense_block():
    block = ResidualInResidualDenseBlock()
    dummy_input = torch.randn(1, 64, 64, 64)
    output = block(dummy_input)
    assert dummy_input.shape == output.shape, "RIRDB should maintain the shape of its input"


def test_generator():
    gen = Generator()
    dummy_input = torch.randn(1, 3, 64, 64)
    output = gen(dummy_input)
    expected_shape = torch.Size([1, 3, 256, 256])
    assert output.shape == expected_shape, "Generator should upscale input image correctly"


def test_ilia_real_esrgan_predict():
    device = 'cpu'
    model_path = 'models/ilia_real_esrgan.pth'
    download_model_from_google_drive('1wv6433R7yAgDblofyo-vTSi3ZmQ7oW_J', model_path)
    model = IliaRealESRGAN(device=device, model_path=model_path)
    input_image = Image.new('RGB', (128, 128), color='red')
    output_image = model.predict(input_image)
    assert output_image.size == (256, 256), "Model should resize and process the image correctly"
