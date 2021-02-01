import torch
import numpy as np
from utility import runner
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
from models.hrnet import hrnet
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



    parser.add_argument(
        "--debug",
        action="store_true",
        help=" ",
    )

    args = parser.parse_args()

    person = Image.open(args.image)
    person = np.array(person)
    image = person.copy()

    bg = Image.open(args.bg_image)
    bg = np.array(bg)
    w, h, _ = image.shape      
    # print(image.shape)
    ww, hh, cc = bg.shape
    # print(bg.shape)
    if ww < w or hh < h:
        bg = cv2.resize(bg, (h, w))
        # print(bg.shape)
    
    model = hrnet(2)
    model.load_state_dict(
        torch.load(args.weights, map_location=torch.device("cpu"))["state_dict"]
    )
    model.eval()

    if args.debug:
        plt.imshow(image)
        plt.show()
        
    with torch.no_grad():
        
        if not ww < w or hh < h:
            yy = ww - w
            xx = hh - h
            yy = int((abs(yy)+yy)/2)
            xx = int((abs(xx)+xx)/2)
            ori_image = np.pad(image, ((yy//2 , yy - yy//2),(xx//2,xx - xx//2),(0,0)), 'constant',  
                    constant_values= 0 ) 
        else:
            ori_image = image.copy()
        # print(ori_image.shape)
        # print(image.shape)
        # ori_image[xx : xx + w, yy : yy + h, :] = image


        prediction = runner(person, model)
        prediction = Image.fromarray(prediction)
        if args.debug:
            plt.imshow(prediction)
            plt.show()
        seg = np.zeros_like(image)
        seg[:,:,0] = prediction

        seg[:,:,1] = prediction
        seg[:,:,2] = prediction 
        
        if not ww < w or hh < h:
                seg = np.pad(seg, ((yy//2 , yy - yy//2),(xx//2,xx - xx//2),(0,0)), 'constant',  
                    constant_values= 0 ) 

        # print(seg.shape)
        result = np.where(seg, ori_image, bg)
        if args.debug:
            plt.imshow(result)
            plt.show()
        final_image = Image.fromarray(result, "RGB")


        final_image.save(f"{args.output_dir}/final.png")  # for Separate Use




if __name__ == "__main__":
    process()
