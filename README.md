# Phase_structure_recognition
You can use OpenMMLab PlayGround, which combined the Segment Anything Model (SAM) with Label-Studio, to semi-automated annotate your electron microscopy images, combine the Segment Anything Model (SAM) with Label-Studio. Just follow the steps below. Users could realize quick marking of images by two methods:Point2Label and Bbox2Label. With Point2Label, users only need to click a point within the object's area to obtain the object's mask and bounding box annotations. With Bbox2Label, users simply annotate the object's bounding box to generate the object's mask. Community users can learn from these methods to improve the efficiency of data annotation.
## Environment Setup
Create a virtual environment by your terminal
```
conda create -n rtmdet-sam python=3.9 -y
conda activate rtmdet-sam
```
PS: If you are unable to use the git command in a Conda environment, you can install git by following the commands below.
```
conda install git
```
