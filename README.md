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

<table>
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI™ Tutorials</h1>
   </td>
 </tr>
 <tr>
 <td align="center"><h3>MNIST and CIFAR-10 Classification using Vitis AI™ 3.5 and TensorFlow 2.16</h3>
 </td>
 </tr>
</table>

### Current status

+ Tested with Vitis AI™ 3.5 and TensorFlow 2.16
+ Tested on the following platforms: ZCU102, Kria KV260


## Introduction

This repository update of the official [AMD Vitis AI™ tutorial](https://github.com/Xilinx/Vitis-AI-Tutorials/blob/1.4/Design_Tutorials/02-MNIST_classification_tf) that guides users through the Vitis AI™ development flow (using Tensorflow), accelerating convolutional neural networks (CNNs) and deploying them on AMD development platforms for inference (prediction). The updates bring the tools to the latest version (2024) and add support for CIFAR-10 dataset classification.

The provided scripts are written in Bash and Python and use VART runtime.

The updated tutorial is divided into the following 3 + 1 steps:

+ Setup of the development environment and definition of environment variables;
+ Definition, training, quantization, and evaluation of a small CNN using TensorFlow 2.16;
+ Compilation of the quantized model to obtain the .xmodel file executable on the DPU accelerator;
+ Deployment of the compiled model on ZCU102 and/or Kria KV260 and execution of the application that handles inference.

Prerequisites:

Familiarity with Python3, TensorFlow, and basic machine learning principles.


## Shell Scripts in this Tutorial

+ ``0_setenv.sh`` : Sets necessary environment variables. You can edit most variables to configure the environment for your own requirements. It is recommended to leave CALIB_IMAGES set to 1000 for proper quantization calibration.

+ ``1_train.sh``: Runs training, quantization (``vai_q_tensorflow quantize``) and evaluation of the network, saveing the trained and quantized model as .h5 file.

+ ``2_compile.sh``: Compiles the quantized model into a .xmodel file for the Kria KV260 platform and for the ZCU102 evaluation board using the `vai_c_tensorflow`` command.

+ ``3_make target.sh``: Copies the .xmodel and images to the ``./build/target_kv260`` folder and to the ``./build/target_zcu102`` folder for use with the Kria KV260 platform and for use with the ZCU102 evaluation board.


## Datasets: MNIST and CIFAR-10 

The MNIST handwritten digits dataset is a publicly available dataset containing 70k 28x28 8-bit grayscale images. The complete dataset of 70k images is divided into 60k images for training and 10k images for validation. It's considered the 'hello world' of Machine Learning.

![mnist](./files/img/mnist.png?raw=true "Example MNIST images")

The CIFAR-10 dataset is a widely used dataset consisting of 60k 32x32 color images divided into 10 different classes, each of which is represented by 6k images. The dataset is split into 50k training images and 10k test images. Widely used for benchmarking in computer vision and Machine Learning.

![cifar10](./files/img/cifar10.png?raw=true "Example CIFAR-10 images")


## The Convolution Neural Network

The updated flow includes two network architectures:

+ The original architecture for MNIST.
+ A slightly more complex architecture for CIFAR-10.

Both networks are described in the ``customcnn.py`` script.


## Image pre-processing

The images undergo simple pre-processing before being used for training, evaluation, and quantization calibration. All images are normalized to bring pixel values into the range of 0 to 1 by dividing by 255.

## Setup the host machine

The host machine must meet several requirements:

+ Ubuntu 20.04 or later
+ At least 100GB of free disk space
+ A CPU or GPU version of the Vitis AI™ docker

NVIDIA GPU (<= CUDA 11.8) is recommended.

For detailed system requirements and installation instructions, refer to the [Vitis AI™ documentation](https://xilinx.github.io/Vitis-AI/3.5/html/docs/reference/system_requirements.html).

Step 1: Install Docker Engine on Ubuntu

1. Uninstall any existing Docker versions:

```shell
"for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done"
```

2. Set up Docker's apt repository:

```shell
# Add Docker's official GPG key
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources
echo \
"deb [arch=$(dpkg --print-architecture) \
signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
sudo apt-get update
```

3. Install the Docker packages:

```shell
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

4. Verify Docker Engine installation:

```shell
sudo docker run hello-world
```

For more information, refer to the [Docker installation guide](https://docs.docker.com/engine/install/ubuntu/).

Step 2: Install CUDA

The following steps referred to hosts with NVIDIA GPU, for a CPU or ROCm installation refer to the [Vitis AI™ documentation](https://xilinx.github.io/Vitis-AI/3.5/html/docs/reference/system_requirements.html).

For NVIDIA GPUs, ensure NVIDIA drivers and the Container Toolkit are correctly installed:

1. Install NVIDIA drivers:

```shell
sudo ubuntu-drivers install
```

2. Configure the repository:

```shell
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update

```

3. Install the NVIDIA Container Toolkit packages:

```shell
sudo apt-get install -y nvidia-container-toolkit
```

4. Verify the installation with the ``nvidia-smi`` command:

```shell
nvidia-smi
```

The output should indicate the successful activation of the driver and installation of CUDA. It should appear similar to the below:

```shell
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
```

For more information, see the [NVIDIA drivers installation guide](https://ubuntu.com/server/docs/nvidia-drivers-installation) and [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.1/install-guide.html).


Step 3: Vitis AI installation

1. Clone the Vitis AI™ repository:

```shell
git clone https://github.com/Xilinx/Vitis-AI
```

2. Build the Vitis AI™ container:

```shell
# Navigate to the docker subdirectory in the Vitis AI install path
cd <Vitis-AI install path>/Vitis-AI/docker

# Build the Tensorflow2 GPU container
sudo ./docker_build.sh -t gpu -f tf2
```

For troubleshooting and more information, refer to the [Vitis AI documentation](https://xilinx.github.io/Vitis-AI/3.5/html/index.html).


## Implementation

Clone or download this GitHub repository to your local machine where you have installed the necessary tools.

```shell
git clone https://github.com/al3monni/MNIST_CIFAR10_classification_tf2.git
```

Open terminal, cd into the ``repository`` folder, then into the ``files`` folder. Start the Vitis AI™ docker.

```shell
# navigate to tutorial folder
cd <path_to_tutorial>/files

# to start GPU docker
./docker_run.sh xilinx/vitis-ai-gpu:latest
```

The docker container will start and you should see something like this in the terminal:

```shell
INSERT OUTPUT HERE
```

The complete tools flow can be run just by executing the `source run_all.sh` command, or by running each of the steps in order (from `0_setenv.sh` to `3_make_target.sh`):

```shell
$ source ./0_setenv.sh
$ source ./1_train.sh
$ source ./2_compile.sh
$ source ./3_make_target.sh
```

## Running the Application on the Kria KV260 platform and ZCU102 board

The `7_make_target.sh` script copies all the required files for running on the Kria KV260 platform and the ZCU102 board into the `files/build/target_kv260` and `files/build/target_zcu102` folders, respectively. Additionally, it copies the test set images to `files/build/target_kv260/images` and `files/build/target_zcu102/images` - the application code will preprocess and classify these images. The entire target_kv260 folder needs to be copied to the KV260 SD card, and the entire target_zcu102 folder needs to be copied to the ZCU102 SD card.

You can directly copy the entire `files/build/target_zcu102` and `files/build/target_kv260` folders to the `/home/root` folder of the respective flashed SD cards. Connect the flashed SD card to your host machine, and when it is recognized you will see two volumes, BOOT and ROOTFS. Navigate into the ROOTFS and then into the /home folder.  Make the ./root folder writeable by issuing the command ``sudo chmod -R 777 root`` and then copy the entire `files/build/target_zcu102` or `files/build/target_kv260` folder from the host machine to the `/home/root` folder on the respective SD card. Then unmount both the BOOT and ROOTFS volumes from the host machine and then eject the SD Card.

After this step plug the SD card into KV260 platform or ZCU102 board and then connect to the platform/board through UART or SSH connection. You can start the application by navigating into the target_kv260 or target_zcu102 folder (`cd target_zcu102` or `cd target_kv260`) and then issuing the command ``python3 app_mt.py``. The application starts, and after a few seconds, shows the throughput (in frames/sec) and the accuracy:

```shell
root@xilinx-zcu102-2021_1:~ cd target_zcu102
root@xilinx-zcu102-2021_1:~/target_zcu102# python3 app_mt.py
Command line options:
 --image_dir :  images
 --threads   :  1
 --model     :  model_dir/customcnn.xmodel
Pre-processing 10000 images...
Starting 1 threads...
Throughput=3644.82 fps, total frames = 10000, time=2.7436 seconds
Correct:9857, Wrong:143, Accuracy:0.9857
```

For better throughput, the number of threads can be increased by using the `-t` option. For example, to execute with four threads:

```shell
root@xilinx-zcu102-2021_1:~/target_zcu102# python3 app_mt.py -t 4
Command line options:
 --image_dir :  images
 --threads   :  4
 --model     :  model_dir/customcnn.xmodel
Pre-processing 10000 images...
Starting 4 threads...
Throughput=5224.95 fps, total frames = 10000, time=1.9139 seconds
Correct:9857, Wrong:143, Accuracy:0.9857
```

</hr>
<p align="center"><sup>Copyright&copy; 2020-2022 Xilinx, Updated by Alessandro Monni, UNISS</sup></p>
