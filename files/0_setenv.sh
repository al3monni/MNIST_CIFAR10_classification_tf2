#!/bin/bash

# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Mark Harvey

# Updated by: Dott.Alessandro Monni, UNISS

#-----------------FOLDERS-----------------

export BUILD=./build 
export LOG=${BUILD}/logs

export COMPILE_KV260=${BUILD}/compile_kv260
export COMPILE_ZCU102=${BUILD}/compile_zcu102

export TARGET_KV260=${BUILD}/target_kv260
export TARGET_ZCU102=${BUILD}/target_zcu102

export APP=./application

#---------REMOVE-AND-MAKE-FOLDERS---------

rm -rf ${BUILD}
mkdir -p ${BUILD}
rm -rf ${LOG}
mkdir -p ${LOG}

#-------------GRAPH-FILENAMES-------------

export FLOAT_MODEL_FILENAME=float.h5
export QUANT_MODEL_FILENAME=quantized.h5

#--------------LOG-FILENAME---------------

export TRAIN_LOG=train.log
export COMP_LOG_KV260=compile_kv260.log
export COMP_LOG_ZCU102=compile_zcu102.log

#---------TRAINING-PARAMETERS-------------

export EPOCHS=10
export BATCHSIZE=64

#-------------GPU-PARAMETERS--------------

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="0"

export TF_FORCE_GPU_ALLOW_GROWTH=true

#-----------NETWORK-PARAMETERS------------

export INPUT_HEIGHT=28   #MNIST height  #32	#CIFAR10 height
export INPUT_WIDTH=28    #MNIST width   #32 #CIFAR10 width
export INPUT_CHAN=1      #MNIST chan    #3  #CIFAR10 chan

export NET_NAME=customcnn

#-----------CALIB-IMAGE-NUMBER------------

export CALIB_IMAGES=1000
