# Electron Dose Point Identification and Tracking Methodology Repository

> **STEM image Point Identification and Detect Methodology Repository.**

## 📌 Repository Overview

This repository focuses on researching and developing methodologies for identifying and tracking points in electron fields, specifically targeting both low and high electron doses. Our aim is to provide reliable and efficient methodologies to address the requirements under various electron dose conditions.

## ✨ Key Features

- 🔍 Specialized in point identification and tracking for both low and high electron doses.
- 🛠 Offers a rich collection of methodologies and algorithm implementations to cater to different scenarios.
- 🤝 Open-source, welcoming contributions from the community for further enhancements and improvements.

## 🚀 Getting Started

To start working with this repository and utilize the methodologies effectively, follow the steps outlined below:

### 0. Environment Setup
Before diving into the methodologies, it's crucial to set up your working environment to ensure compatibility and proper functioning of all tools.
Key steps:
🛠 Install Required Packages: Ensure your Python environment is ready by installing all required packages using: pip install -r requirements.txt
This command will install all the necessary dependencies as listed in the requirements.txt file. 

### 1. Dataset Organization

Before applying the methodologies, it's essential to organize the dataset properly. Refer to the detailed instructions provided in [`stem_dataset/dataset_prepare.md`](./stem_dataset/dataset_prepare.md) for comprehensive information on how to prepare and structure the dataset.

#### Key steps:

- 📦 **Data Collection**: Gather the electron dose dataset from reliable sources or experiments.
- 🧹 **Data Preprocessing**: Clean, format, and preprocess the data to make it suitable for further analysis.
- 🏷 **Labeling**: Annotate the dataset with detailed labels and relevant information, which is crucial for training and evaluation.

### 2. Detailed Labeling Instructions

For a thorough guide on labeling content and best practices, consult the [`dataset_prepare.md`](./stem_dataset/dataset_prepare.md) document. It covers the specifics of labeling data for this project, ensuring consistency and accuracy in the dataset.

📝 **Note**: After completing the above steps, ensure your data is placed into the [`stem_dataset`](./stem_dataset/) in the specified format.

### 3. Train Steps

#### 3.1 Preprocess Model Training 

To begin training the preprocess model:

- **Config Files Modification**: Navigate to the preprocess model's folder and access the [`configs`](./preprocess_model/configs) sub-directory. Within this directory, you'll find:

  - [`option.py`](./preprocess_model/configs/option.py): Modify parameters specific to the SRmodel, such as adjusting the depth of the network and the channels used for feature extraction.
  - `vae.yaml`: Adjust hyperparameters related to the model's training, like batch size, learning rate (`lr`), `manual_seed`, etc.

- **Training**: Run the [`ensemble_model_train.py`](./ensemble_model_train.py) to start training the preprocess model.

#### 3.2 Molecule Structure Recognition Model Training

1. **Understanding Training Parameters**: Navigate to the `structure_recognition` directory and inspect the `train.py` script, which contains details on all the model's training parameters.
2. **Dataset Preparation**: To train on your own dataset, ensure it adheres to the format in the `faster_rcnn_stem_dataset` directory.

### Model Weights Download and Testing Instructions

You can download the corresponding weights for the models and run them using the `test_image_ml_framework.py` script. Below are the links to download the weights for the specific models:

- For Faster R-CNN model weights, download from [this link](https://drive.google.com/drive/folders/1P60SL8MKod3kxL4OmkqVlzCfrLXydZgl?usp=sharing).
- For DIVAESR model weights, download from [this link](https://drive.google.com/drive/folders/1TG5OyvcmJ9sTD_H7OAC8U3ue36ZF8V2u?usp=sharing).

Please create corresponding folders to store the model weights after downloading them. Then, update the paths to the model weights in the `test_image_ml_framework.py` script to reflect where you have stored the weights on your local machine.

#### Steps to Run the Test:

1. Download the model weights using the links provided above.
2. Create folders for each model's weights on your local machine.
3. Update the `test_image_ml_framework.py` script with the correct paths to the model weights.
4. Run the `test_image_ml_framework.py` script to test the models with the downloaded weights.

This will allow you to test the models with the provided weights effectively.

## 🤲 How to Contribute

We heartily welcome all individuals interested in electron dose point identification and tracking methodologies to contribute. Engage by:

1. 🐞 Submitting Issues: Share thoughts, identify bugs, or offer suggestions.
2. 🔄 Submitting Pull Requests: Collaborate on enhancing our methodologies and implementations.
3. 🌍 Sharing: Help spread the word and engage more enthusiasts.

🙏 **Thank you** for your interest and support in our repository!
