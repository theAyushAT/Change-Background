
# Change-Background
Change background of image using semantic segmentation


[Click ](http://getplates.ml/backround-replacer) and change your background now
=======
## Running script

### Weights

Download weights from here:

https://drive.google.com/file/d/1VNYZ1X5bnIZFIVIgNi9JwqCNIvioxexe/view?usp=sharing

### Run 

python main.py --image --bg_image --weights --debug --output_dir



### Input 

<img src="https://github.com/theAyushAT/Background-Change/blob/main/demo_images/background.jpg" width="320.0" height= "213.3"> <img src= "https://github.com/theAyushAT/Background-Change/blob/main/demo_images/image1.jpg" width= "325" height= "487.5">


## Output
<img src="https://github.com/theAyushAT/Background-Change/blob/main/demo_images/final1.png" width= "325" height= "487.5">



### Example

python main.py --image demo_images/image1.jpg --bg_image demo_images/background.jpg --weights hrnetv2_hrnet18_person_dataset_120.pth --debug



