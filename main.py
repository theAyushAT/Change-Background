import torch
import numpy as np
from utility import runner
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
from models.hrnet import hrnet
import argparse


def main():
    
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

    person = Image.open(args.image) # reads person image
    person = np.array(person)
    image = person.copy()

    bg = Image.open(args.bg_image) # reads background image
    bg = np.array(bg)
    w, h, _ = image.shape      

    ww, hh, _ = bg.shape

    # for debugging shows images
    if args.debug:
        plt.imshow(image)
        plt.show()
        plt.imshow(bg)
        plt.show()
        
    # comparison between size
    if ww < w or hh < h:
        bg = cv2.resize(bg, (h, w)) # if size of background < size of person image
        #bg is resized to person image

    #mocel is created
    model = hrnet(2)
    model.load_state_dict(
        torch.load(args.weights, map_location=torch.device("cpu"))["state_dict"]
    ) # model is loaded
    model.eval()

    with torch.no_grad(): # gradients are off
        
        #padding of image of person , so that both background image and person image are same

        yy = ww - w
        xx = hh - h
        yy = int((abs(yy)+yy)/2)
        xx = int((abs(xx)+xx)/2)
        ori_image = np.pad(image, ((yy//2 , yy - yy//2),(xx//2,xx - xx//2),(0,0)), 'constant',  
                constant_values= 0 ) 


        prediction = runner(person, model) # output from semantic segmentation model
        prediction = Image.fromarray(prediction) # from numpy convert to PIL format
        
        if args.debug:
            plt.imshow(prediction)
            plt.show()
        # prediction is converted to 3D array
        seg = np.zeros_like(image)
        seg[:,:,0] = prediction

        seg[:,:,1] = prediction
        seg[:,:,2] = prediction 
        
        seg = np.pad(seg, ((yy//2 , yy - yy//2),(xx//2,xx - xx//2),(0,0)), 'constant',  
            constant_values= 0 ) 

        result = np.where(seg, ori_image, bg) # array is made by keeping person image pixel where person is present else background image pixels 
        if args.debug:
            plt.imshow(result)
            plt.show()
        final_image = Image.fromarray(result, "RGB")


        final_image.save(f"{args.output_dir}/final.png")  # for Separate Use




if __name__ == "__main__":
    main()
