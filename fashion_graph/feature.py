import numpy as np
from typing import *
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.color import gray2rgb, rgba2rgb


class FeatureExtractor:
    def __init__(self, device):
        self.model = models.resnet50(pretrained=True)
        self.device = device
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def preprocess(image: np.ndarray):
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if image.shape[2] == 4:
            image = rgba2rgb(image)

        image = resize(image, (256, 256))
        image = img_as_ubyte(image)
        return image

    def get_feature(self, image):
        """
        Image must be tensor or np.ndarray
        """
        image = self.transform(image)
        image = image.unsqueeze_(0)
        image = image.to(self.device)
        out = self.model(image)
        return out.squeeze().detach().cpu().numpy()
