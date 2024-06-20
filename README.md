<!--

Copyright 2020-2022 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Author: Mark Harvey, Xilinx Inc

Updated by Dott.Alessandro Monni, UNISS

-->


### Current status

+ Tested with Vitis AI&trade; 3.5 and TensorFlow 2.16
+ Tested on the following platforms: ZCU102, Kria KV260


## Introduction

This repository update of the official AMD Vitis AI™ tutorial designed to guide new users through the Vitis AI development flow (using Tensorflow) that accelerate convolutional neural networks (CNNs) and deploy them on AMD development platforms for inference (prediction). The updates bring the tools to the latest version (2024), making the flow simple and faster and enabling classification to the CIFAR-10 dataset.

The provided scripts are written in Bash and Python and use VART runtime.

The updated tutorial is divided into the following 3 + 1 steps:

+ Setup of the development environment and definition of environment variables;
+ Definition, training, quantization, and evaluation of a small CNN using TensorFlow 2.16;
+ Compilation of the quantized model to obtain the .xmodel file executable on the DPU accelerator;
+ Deployment of the compiled model on ZCU102 and/or Kria KV260 and execution of the application that handles inference.

This tutorial assumes the user is familiar with Python3, TensorFlow and has some knowledge of machine learning principles.


## Shell Scripts in this Tutorial

+ ``0_setenv.sh`` : Sets all the necessary environment variables used by the other scripts. You can edit most variables to configure the environment for their own requirements.

It is highly recommended to leave the ``CALIB_IMAGES`` variable set to 1000 because it is the minimum recommended number of images for calibration of the quantization.

+ ``1_train.sh``: Runs training, quantization (``vai_q_tensorflow quantize``) and evaluation of the network. It saves the trained and quantized model as .h5 file extension.

+ ``2_compile_kv260.sh``: Launches the ``vai_c_tensorflow`` command to compile the quantized model into a .xmodel file for the Kria KV260 platform.
+ ``2_compile_zcu102.sh``: Launches the ``vai_c_tensorflow`` command to compile the quantized model into an .xmodel file for the ZCU102 evaluation board.

+ ``3_make target_kv260.sh``: Copies the .xmodel and images to the ``./build/target_kv260`` folder ready for use with the Kria KV260 platform.
+ ``3_make target_zcu102.sh``: Copies the .xmodel and images to the ``./build/target_zcu102`` folder ready to be copied to the ZCU102 evaluation board's SD card.


## The MNIST and CIFAR-10 Datasets

The MNIST handwritten digits dataset is a publicly available dataset containing 70k 28x28 8-bit grayscale images. The complete dataset of 70k images is divided into 60k images for training and 10k images for validation. The dataset is considered to be the 'hello world' of Machine Learning (ML).

![mnist](./files/img/mnist.png?raw=true "Example MNIST images")

The CIFAR-10 dataset is a widely used dataset consisting of 60k 32x32 color images divided into 10 different classes, each of which is represented by 6k images. The dataset is split into 50k training images and 10k test images, providing a comprehensive dataset for training and evaluating ML models. The CIFAR-10 dataset is often used as a benchmark in the field of Computer Vision and ML, making it an excellent resource for both beginners and experienced practitioners.

![cifar10](./files/img/cifar10.png?raw=true "Example CIFAR-10 images")


## The Convolution Neural Network

The introduced changes to update the flow include two network architectures, the original architecture used to classify MNIST and an additional slightly more complex architecture to classify CIFAR-10. Both networks are described in the customcnn.py Python script.


## Image pre-processing

All images undergo simple pre-processing before being used for training, evaluation, and quantization calibration. The images are normalized to bring all pixel values into the range of 0 to 1 by dividing them by 255.


## Setup the host machine

La host machine deve soddisfare svariati requisiti:

Ubuntu 20.04 o versioni successive
Almeno 100GB di spazio libero su disco
Una versione CPU o GPU del docker di Vitis AI

Per maggiori informazioni consultare Vitis AI Host (Developer) Machine Requirements e Host Installation Instructions — Vitis™ AI 3.5 documentation.

È consigliata una GPU, preferenzialmente Nvidia (<= CUDA 11.8) o in alternativa AMD (ROCm), tuttavia, l’intero flusso può essere portato a termine senza.

This tutorial assumes the user is familiar with Python3, TensorFlow and has some knowledge of machine learning principles.

Step 1: Install Docker Engine on Ubuntu

+ Uninstall all versions :

→ "for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done"

+ Install docker engine using apt repository (or with your favourite method) :
  
1. Set up Docker's apt repository.

→ # Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

→ # Add the repository to Apt sources:
echo \
"deb [arch=$(dpkg --print-architecture) \
signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
sudo apt-get update

2. Install the Docker packages.

→ sudo apt-get install \
docker-ce \
docker-ce-cli \
containerd.io \
docker-buildx-plugin \
docker-compose-plugin

3. Verify Docker Engine installation.

→ sudo docker run hello-world

Per maggiori informazioni consultare la guida di installazione di Docker Engine.

Step 2: Install CUDA

The following steps referred to hosts with NVIDIA GPU, per un installazione cpu-only o ROCm consultare la guida di installazione di Vitis AI.

Per le GPU NVIDIA è necessario installare correttamente i driver NVIDIA e il Toolkit Container:

→ sudo ubuntu-drivers install

+ Configure the repository:

→ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
&& \
sudo apt-get update

+ Install the NVIDIA Container Toolkit packages:

→ sudo apt-get install -y nvidia-container-toolkit

+ Verificare l’installazione con il comando nvidia-smi:

→ nvidia-smi

The output should appear similar to the below, indicating the activation of the driver, and the successful installation of CUDA:

Wed Jun 19 12:13:28 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.171.04             Driver Version: 535.171.04   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4060        Off | 00000000:01:00.0  On |                  N/A |
|  0%   41C    P8              N/A / 115W |    268MiB /  8188MiB |      1%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1628      G   /usr/lib/xorg/Xorg                          112MiB |
|    0   N/A  N/A      1878      G   /usr/bin/gnome-shell                        107MiB |
|    0   N/A  N/A      3291      G   /usr/bin/gnome-text-editor                   14MiB |
|    0   N/A  N/A      4459      G   /usr/bin/nautilus                            11MiB |
+---------------------------------------------------------------------------------------+

Per maggiori informazioni consultare: NVIDIA drivers installation | Ubuntu e Installing the NVIDIA Container Toolkit

In alternativa consultare la guida di installazione CUDA.


Step 3: Vitis AI installation

+ Clone the repository :

→ git clone https://github.com/Xilinx/Vitis-AI

+ Build VITIS AI container:

1. Navigate to the docker subdirectory in the Vitis AI install path:

→ cd <Vitis-AI install path>/Vitis-AI/docker

2. Build the Tensorflow2 GPU container:

→ ./docker_build.sh -t gpu -f tf2

Per qualsiasi problema e maggiori informazioni consultare la guida di installazione di Vitis AI.
