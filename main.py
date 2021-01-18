import torch
import numpy as np
from utility import runner
from PIL import Image
import os
from preload import preloader
import cv2
import matplotlib.pyplot as plt
from models import hrnet
import argparse


def process():
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        default="./",
        help="path to foreground image(single person image)",
    )

    parser.add_argument(
        "--bg_image",
        type=str,
        required=True,
        default="./",
        help="path to background image",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        default="./",
        help="Path to weights for which inference needs to be done",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="path to save output image",
    )



    # parser.add_argument(
    #     "--is_video",
    #     action="store_true",
    #     help="If true path to image will be video ",
    # )

    args = parser.parse_args()

    person = Image.open(args.image)
    person = np.array(person)
    image = person.copy()

    bg = Image.open(args.bg_image)
    bg = np.array(bg)
    w, h, _ = person.shape
    # print(person.shape)
    ww, hh, cc = bg.shape
    # print(bg.shape)
    if ww < w and hh < h:
        bg = cv2.resize(bg, (h, w))
        ww, hh, cc = bg.shape
        
        
    model = hrnet()
    model.load_state_dict(
        torch.load(seg_weights, map_location=torch.device("cpu"))["state_dict"]
    )
    model.eval()

    with torch.no_grad():

        ori_image = np.zeros_like(bg)

        yy = (ww - w) // 2
        xx = (hh - h) // 2

        ori_image[xx : xx + w, yy : yy + h, :] = image


        prediction = runner(person, model)
        prediction = Image.fromarray(prediction)
        seg = np.zeros((ww, hh, 3))

        seg[xx : xx + w, yy : yy + h, 0] = prediction
        seg[xx : xx + w, yy : yy + h, 1] = prediction
        seg[xx : xx + w, yy : yy + h, 2] = prediction

        result = np.where(seg, ori_image, bg)
        if debug:
            plt.imshow(result)
            plt.show()
        final_image = Image.fromarray(result, "RGB")


        final_image.save(f"{args.output_dir}/final.png")  # for Separate Use

    return img



if __name__ == "__main__":
    process()
