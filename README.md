# ITNPAI1_Vehicles_Detection
---
# 1. **Problem definition** 

This project presents deep learning based solution for vehicles detection including cars, motorcycles, buses, and trucks from wild images at street-level. The dataset is collected from two cities which are Cairo and Stirling.
---
# 2. **Dataset creation**
202 images from Cairo and 200 images from Stirling are collected using **Mapillary API**. The first step was collecting sequence keys for each city and save them in seperate text file. Then, this [notebook](https://github.com/hedayaahmed/ITNPAI1_Vehicles_Detection/blob/main/Mapillary.ipynb) was used to read these keys and send a request to **Mapillary API** to get response and download these images.

Cairo dataset includes 809 cars, 49 motorcycles, 15 buses, and 57 trucks, while Stirling dataset contains 548 cars, 2 motorcycles, 7 buses, and 19 trucks.
The dataset was annotated using Computer Vision Annotation Tool "CVAT". Then, it is exported in Yolo format and converted to PASCAL VOC format using this [notebook](https://github.com/hedayaahmed/ITNPAI1_Vehicles_Detection/blob/main/Yolo%20to%20PASCAL.ipynb). The annotation consists of bounding boxes "xmin, ymin, xmax, ymax", object class, and filter label for the repeated, low quality, or empty images to be removed from the final dataset.
---
# 4. **Dataloader**
After some image preprocessing such as converting images from BGR to RGB, normalization, and resizing, the dataset splitted to training, validation, and testing. After that, data augmentation techniques were applied to increase number of images. Finally, the dataset was loaded into tensor for the next model training step.

Some scripts of vision package were modified because it was necessary to be edited to be able to use generated statitics and matrics from training and validation functions in order to visualize losses and Mean average precision "mAP" matrics.

These scripts are:
- [coco_eval.py](https://github.com/hedayaahmed/ITNPAI1_Vehicles_Detection/blob/main/coco_eval.py)
- [coco_utils.py](https://github.com/hedayaahmed/ITNPAI1_Vehicles_Detection/blob/main/coco_utils.py)
- [engine.py](https://github.com/hedayaahmed/ITNPAI1_Vehicles_Detection/blob/main/engine.py)
- [transforms.py](https://github.com/hedayaahmed/ITNPAI1_Vehicles_Detection/blob/main/transforms.py)
- [utils.py](https://github.com/hedayaahmed/ITNPAI1_Vehicles_Detection/blob/main/utils.py)

![Uploaded scripts](https://drive.google.com/uc?export=view&id=1lDWB19o5pzSVxLJjtmJWYpzbLicR00Sq)
---
# 5. **Proposed solution** 

The proposed solution is Faster R-CNN with ResNet50 as a backbone. In this step of the pipeline, the model was trained on the training set for specified number of epochs using different hyperparamters, and it was evaluated on the validation set. During this, the loss, Mean average precision, and Mean average recall were calculated. Finally, two postprocessing step were implemented, which are appling Non Maximum Suppression "NMS", and converting images tensor to PIL images.
---
# 6. **Experimental tests and evaluations** 

Here is the experiments of training, testing and evaluating the proposed solution:
- *E1* - Use half (50%) of the samples from each dataset for training and leave the other half for testing (50%).
- *E2* - Test the predictive ability of your models using half of each dataset intended for testing. **The model trained in city A must be tested in city A. The model trained in city B must be tested in city B.**
- *E3* - Testing the models crossing datasets. Now, training is on one city and testing in the other. **The model trained in city A must be tested in city B. The model trained in city B must be tested in city A.**
