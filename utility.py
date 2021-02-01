import numpy as np
import torch
import time
import os
import sys
import cv2
from torchvision import transforms

transformation = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


def normalize(image):
    return transforms.Normalize(mean=[0.473408, 0.44432889, 0.42011778], std=[0.23041105, 0.22339764, 0.22698703])(
        transformation(image)
    )

def preprocess_image(image):

    image = normalize(image)
    return torch.unsqueeze(image, dim=0)


def runner(frame, model):

    current_path = os.path.dirname(os.path.abspath(__file__))

    w, h, _ = frame.shape
    import matplotlib.pyplot as plt

    image = preprocess_image(frame)

    with torch.no_grad():
        prediction = model(image, (w, h))

    prediction = (
        torch.argmax(prediction["output"][0], dim=0)
        .cpu()
        .squeeze(dim=0)
        .numpy()
        .astype(np.uint8)
    )
    # print(prediction.shape)
    # print(w,h)
    # plt.imshow(prediction)
    # plt.show()
    return prediction
