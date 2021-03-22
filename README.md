# Background-Change
Change background of image using semantic segmentation

[Click ](http://getplates.ml/backround-replacer) and change your background now


### Input 

<img src="https://github.com/theAyushAT/Background-Change/blob/main/demo_images/background.jpg" width="320.0" height= "213.3"> <img src= "https://github.com/theAyushAT/Background-Change/blob/main/demo_images/image1.jpg" width= "325" height= "487.5">


## Output
<img src="https://github.com/theAyushAT/Background-Change/blob/main/demo_images/final1.png" width= "325" height= "487.5">

## Running script

python main.py --image --bg_image --weights --debug --output_dir

### Example

python main.py --image demo_images/image1.jpg --bg_image demo_images/background.jpg --weights hrnetv2_hrnet18_person_dataset_120.pth --debug



