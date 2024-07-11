import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channel=64, inc_channel=32, beta=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, inc_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channel + inc_channel, inc_channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channel + 2 * inc_channel, inc_channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channel + 3 * inc_channel, inc_channel, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channel + 4 * inc_channel, in_channel, 3, 1, 1)
        self.lrelu = nn.LeakyReLU()
        self.b = beta

    def forward(self, x):
        block1 = self.lrelu(self.conv1(x))
        block2 = self.lrelu(self.conv2(torch.cat((block1, x), dim=1)))
        block3 = self.lrelu(self.conv3(torch.cat((block2, block1, x), dim=1)))
        block4 = self.lrelu(self.conv4(torch.cat((block3, block2, block1, x), dim=1)))
        out = self.conv5(torch.cat((block4, block3, block2, block1, x), dim=1))

        return x + self.b * out


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, in_channel=64, out_channel=32, beta=0.2):
        super().__init__()
        self.RDB = ResidualDenseBlock(in_channel, out_channel)
        self.b = beta

    def forward(self, x):
        out = self.RDB(x)
        out = self.RDB(out)
        out = self.RDB(out)

        return x + self.b * out


class Generator(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, noRRDBBlock=23):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)

        RRDB = ResidualInResidualDenseBlock()
        RRDB_layer = []
        for i in range(noRRDBBlock):
            RRDB_layer.append(RRDB)
        self.RRDB_block = nn.Sequential(*RRDB_layer)

        self.RRDB_conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.upconv = nn.Conv2d(64, 64, 3, 1, 1)

        self.out_conv = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        first_conv = self.conv1(x)
        RRDB_full_block = torch.add(self.RRDB_conv2(self.RRDB_block(first_conv)), first_conv)
        upconv_block1 = self.upconv(F.interpolate(RRDB_full_block, scale_factor=2))
        upconv_block2 = self.upconv(F.interpolate(upconv_block1, scale_factor=2))
        out = self.out_conv(upconv_block2)

        return out


class IliaRealESRGAN:
    def __init__(self, device, model_path):
        self.device = device
        self.model = Generator().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def resize_image(self, image):
        width, height = image.size
        if width > 512 or height > 512:
            if width > height:
                new_width = 512
                new_height = int(512 * height / width)
            else:
                new_height = 512
                new_width = int(512 * width / height)
            image = image.resize((new_width, new_height), Image.BICUBIC)
        elif width < 64 or height < 64:
            if width < height:
                new_width = 64
                new_height = int(64 * height / width)
            else:
                new_height = 64
                new_width = int(64 * width / height)
            image = image.resize((new_width, new_height), Image.BICUBIC)
        return image

    def predict(self, input_image):
        input_image = self.resize_image(input_image)
        input_tensor = self.preprocess(input_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            sr_tensor = self.model(input_tensor)
        sr_image = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sr_image = (sr_image * 0.5 + 0.5) * 255.0
        sr_image = sr_image.clip(0, 255).astype('uint8')
        sr_image = Image.fromarray(sr_image)

        original_width, original_height = sr_image.size
        new_width = original_width // 2
        new_height = original_height // 2
        sr_image = sr_image.resize((new_width, new_height), Image.BICUBIC)

        return sr_image
