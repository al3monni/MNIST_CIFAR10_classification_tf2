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

echo -e "\n-----------------------------------------"
echo    " MAKE TARGET STARTED.."
echo -e "-----------------------------------------\n"

# remove previous results
rm -rf ${TARGET_KV260}
mkdir -p ${TARGET_KV260}/model_dir

# remove previous results
rm -rf ${TARGET_ZCU102}
mkdir -p ${TARGET_ZCU102}/model_dir


# copy application to TARGET_KV260 folder, comment this function if you don't need the kria support to dont't waste time
cp ${APP}/*.py ${TARGET_KV260}
echo " Copied application to TARGET_KV260 folder"

# copy application to TARGET_ZCU102 folder, comment this function if you don't need the zcu support to dont't waste time
cp ${APP}/*.py ${TARGET_ZCU102}
echo " Copied application to TARGET_ZCU102 folder"


# copy xmodel to TARGET_KV260 folder, comment this function if you don't need the kria support to dont't waste time
cp ${COMPILE_KV260}/${NET_NAME}.xmodel ${TARGET_KV260}/model_dir/.
echo " Copied xmodel file(s) to TARGET_KV260 folder"

# copy xmodel to TARGET_ZCU102 folder, comment this function if you don't need the zcu support to dont't waste time
cp ${COMPILE_ZCU102}/${NET_NAME}.xmodel ${TARGET_ZCU102}/model_dir/.
echo " Copied xmodel file(s) to TARGET_ZCU102 folder"


# create image files and copy to target folder
mkdir -p ${TARGET_KV260}/images

# create image files and copy to target folder
mkdir -p ${TARGET_ZCU102}/images

# generate test image for inference, comment this function if you don't need the kria support to dont't waste time
python generate_images.py  \
    --dataset=mnist \
    --image_dir=${TARGET_KV260}/images \
    --image_format=jpg \
    --max_images=10000

echo "  Copied images to TARGET_KV260 folder"

# generate test image for inference, comment this function if you don't need the zcu support to dont't waste time
python generate_images.py  \
    --dataset=mnist \
    --image_dir=${TARGET_ZCU102}/images \
    --image_format=jpg \
    --max_images=10000

echo "  Copied images to TARGET_ZCU102 folder"


echo -e "\n-----------------------------------------"
echo    " MAKE TARGET COMPLETED"
echo    "-----------------------------------------"
