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
```shell
conda install git
```

Clone OpenMMLab PlayGround

```shell
git clone https://github.com/open-mmlab/playground
```

If you encounter network errors, try to complete the git clone via ssh, like the following command:

```shell
git clone git@github.com:open-mmlab/playground.git
```

Install PyTorch

```shell
# Linux and Windows CUDA 11.3
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html


# Linux and Windows CPU only
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html

# OSX
pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1

```

Install SAM and download the pre-trained model:

```shell
cd path/to/playground/label_anything
# Before proceeding to the next step in Windows, you need to complete the following command line.
# conda install pycocotools -c conda-forge
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# If you're on a windows machine you can use the following in place of wget
# curl -L -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# For better segmentation results, use the sam_vit_h_4b8939.pth weights
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# download HQ-SAM pretrained model
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth
#wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth
#wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth

# download mobile_sam pretrained model
wget https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt
# or manually download mobile_sam.pt in https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/, and put it into path/to/playground/label_anything

```

PS: If you are using a having trouble with the wget/curl commands, please manually download the target file (copy the URL to a browser or download tool). The same applies to the following instructions.
For example: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Install Label-Studio and label-studio-ml-backend

Currently, label_anything supports three inference models: SAM, HQ-SAM, and mobile_sam. Users can choose according to their own needs, but note that the model and the downloaded weights in the previous step need to correspond. HQ-SAM has higher segmentation quality than SAM. Mobile_sam has faster inference speed and lower memory usage than SAM, and the segmentation effect only slightly decreases. It is recommended to use mobile_sam for CPU inference.

```shell
# sudo apt install libpq-dev python3-dev # Note: If using Label Studio 1.7.2 version, you need to install libpq-dev and python3-dev dependencies.

# Installing label-studio may take some time. If you cannot find the version, please use the official source.
pip install label-studio==1.7.3
pip install label-studio-ml==1.0.9
```

## Start the service

⚠label_anything requires the SAM backend to be enabled and then the web service to be started before the model can be loaded. (a total of two steps are required to start)

1.Start the SAM backend inference service:

```shell
cd path/to/playground/label_anything

# inference on sam
label-studio-ml start sam --port 8003 --with \
  model_name=mobile_sam \
  sam_config=vit_b \
  sam_checkpoint_file=./sam_vit_b_01ec64.pth \
  out_mask=True \
  out_bbox=True \
  device=cuda:0
# device=cuda:0 is for using GPU inference. If you want to use CPU inference, replace cuda:0 with cpu.
# out_poly=True returns the annotation of the bounding polygon.

# inference on HQ-SAM
label-studio-ml start sam --port 8003 --with \
  sam_config=vit_b \
  sam_checkpoint_file=./sam_hq_vit_l.pth \
  out_mask=True \
  out_bbox=True \
  device=cuda:0 
# device=cuda:0 is for using GPU inference. If you want to use CPU inference, replace cuda:0 with cpu.
# out_poly=True returns the annotation of the bounding polygon.

# inference on mobile_sam
label-studio-ml start sam --port 8003 --with \
  model_name=mobile_sam  \
  sam_config=vit_t \
  sam_checkpoint_file=./mobile_sam.pt \
  out_mask=True \
  out_bbox=True \
  device=cpu 
# device=cuda:0 is for using GPU inference. If you want to use CPU inference, replace cuda:0 with cpu.
# out_poly=True returns the annotation of the bounding polygon.

```
PS: In Windows environment, entering the following in Anaconda Powershell Prompt is equivalent to the input above:

```shell
cd path/to/playground/label_anything

$env:sam_config = "vit_b"
$env:sam_checkpoint_file = ".\sam_vit_b_01ec64.pth"
$env:out_mask = "True"
$env:out_bbox = "True"
$env:device = "cuda:0"
# device=cuda:0 is for using GPU inference. If you want to use CPU inference, replace cuda:0 with cpu.
# out_poly=True returns the annotation of the bounding polygon.

label-studio-ml start sam --port 8003 --with `
sam_config=$env:sam_config `
sam_checkpoint_file=$env:sam_checkpoint_file `
out_mask=$env:out_mask `
out_bbox=$env:out_bbox `
device=$env:device

# mobile_sam on windows have not been tested, if you are interesteed in it, please modify the shell script like upper script.
```

![image](https://user-images.githubusercontent.com/25839884/233821553-0030945a-8d83-4416-8edd-373ae9203a63.png)

At this point, the SAM backend inference service has started.

⚠The above terminal window needs to be kept open.

Next, please follow the steps below to configure the use of the back-end reasoning service in the Label-Studio Web system.

2.Now start the Label-Studio web service:

Please create a new terminal window to access the label_anything project path.

```shell
cd path/to/playground/label_anything
```

⚠(Please skip this step if you do not use SAM with vit-h) The inference backend used is SAM's **vit-h**, which requires the following environment variables to be set due to the long loading time of the model, which causes the connection to the backend to time out.

The specific can be set according to the name of the downloaded SAM's weights, such as sam_vit_h_4b8939.pth for vit-h and sam_vit_b_01ec64.pth for vit-b.

```shell
# Linux requires the following commands
export ML_TIMEOUT_SETUP=40
# Windows requires the following commands
set ML_TIMEOUT_SETUP=40
```

Start Label-Studio web service:

```shell
label-studio start
```

![](https://cdn.vansin.top/picgo20230330132913.png)

Open your browser and visit [http://localhost:8080/](http://localhost:8080/) to see the Label-Studio interface.

![](https://cdn.vansin.top/picgo20230330133118.png)

We will register a user and then create an OpenMMLabPlayGround project.
PS: Label-Studio's username and password are stored locally. If you encounter a situation where the browser remembers the password but you are unable to log in, please register again.

![](https://cdn.vansin.top/picgo20230330133333.png)

## Frontend Configuration

### Import images to be annotated:
