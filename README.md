# Assembly_Line_Monitoring
Object Detection project for iotiot.in internship

## Problem Statement: Automated Product Classification and Counting
In modern manufacturing companies, there is a need for efficient and reliable **classification and counting** of packaged products as they pass on a conveyor belt. A well trained object detection model will make this highly efficient and minimize human error.
### Goal:
The goal of the project is to train an object detection model that can be deployed in warehouse conveyor belt systems to efficiently count and classify different types of cartons that pass on it.

### Classes:
The model is expected to identify and classify multiple variations of cartons based on size and color.

## Procedure:

After deciding on the problem statement, the first and most important task is to look for datasets online and/or curating a custom dataset.  
For this project, we will be using this dataset: https://universe.roboflow.com/moyed-chowdhury/mv_train_dataset_2  
We are following this [medium](https://medium.com/analytics-vidhya/training-a-model-for-custom-object-detection-tf-2-x-on-google-colab-4507f2cc6b80) article for reference.

* Downloaded the above dataset with PASCAL_VOC XML annotations

* For preprocessing, ensured that all images are in .jpg file format, all images are relatively good in quality, and all files are serially named.  
Wrote a small pyhton code to automate renaming the image filenames and the corresponding file path in the XML annotations

```py
import os
import xml.etree.ElementTree as ET

path1 = 'C:\\Stuffing\\ImPing\\ObjectDetection\\annotations'
path2 = 'C:\\Stuffing\\ImPing\\ObjectDetection\\images'
annots = os.listdir(path1)
imgs = os.listdir(path2)
filename = 'testname'

for i in range(len(annots)):
    filename = 'image_' + str(i+1)
    filejpg = filename + '.jpg'
    os.rename(f'images\\{imgs[i]}', filejpg)

    mytree = ET.parse(f'annotations\\{annots[i]}')
    root = mytree.getroot()

    for child in root:
        if child.tag == 'filename':
            child.text = filejpg
            child.set('updated', 'yes')
        if child.tag == 'path':
            child.text = filejpg
            child.set('updated', 'yes')

    mytree.write(f'{filename}.xml')
```
* Created google drive folder to mount on the colaboratory.
* Uploaded zip files of the images and the annotations.
* Began the new colab with importing necessary libraries, mounting our drive and cloning the https://github.com/tensorflow/models.git repo into it.
* Next, unzipped the images and the annotations folder into a new /data folder ands made new directories.
* Created the csv files and .pbtxt file. Ran the *generate_tfrecord.py* script to create *train.record* and *test.record* files
* Downloaded pre-trained model checkpoint.
* Next, we had to make changes to the config file based on our specific implementation parameters (changed no. of classes, paths to all the created files like csv file and record files)
* Loaded tenserboard to visalize and log the model training process.
* This completes our pre-training steps. Start model training now by running the given command:
```
!python model_main_tf2.py --pipeline_config_path=/mydrive/customTF2/data/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config --model_dir=/mydrive/customTF2/training --alsologtostderr
```

## Errors encountered:
1. After running the training command, we encountered the following error:
```
attributeerror: module 'tensorflow.python.ops.control_flow_ops' has no attribute 'case'
```
Found a solution online on github community. The error is associated with the naming of certain model files and inconsistent tensorflow versions.  

Resolved by following [this comment](https://github.com/tensorflow/models/issues/11099#issuecomment-1902615454)  

2. After running the training command, we are encountering the following error:
```
tensorflow.python.framework.errors_impl.InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_16_device_/job:localhost/replica:0/task:0/device:CPU:0}} assertion failed: [[0.109177209][0.571202576][0.136075944]] [[0.422468364][0.354430407][0.481012702]]
	 [[{{function_node Assert_1_AssertGuard_false_873}}{{node Assert_1/AssertGuard/Assert}}]]
	 [[MultiDeviceIteratorGetNextFromShard]]
	 [[RemoteCall]] [Op:IteratorGetNext] name:
```
After looking for solutions online, it appears that the error might be due to the coordinates of the bounding box for the annotations of some images.  
The types of errors in this could be:  
	a. Negative values  
	b. Value out of bounds (more than the height/width)  
	c. xmin > xmax or ymin > ymax  

