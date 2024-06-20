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

# train, evaluate & save trained model
run_train() {  
  python -u train.py \
    --input_height      ${INPUT_HEIGHT} \
    --input_width       ${INPUT_WIDTH} \
    --input_chan        ${INPUT_CHAN} \
    --epochs            ${EPOCHS} \
    --batchsize         ${BATCHSIZE} \
    --calib_images      ${CALIB_IMAGES} \
    --gpu               ${CUDA_VISIBLE_DEVICES} \
    --float_model_path  ${BUILD}/${FLOAT_MODEL_FILENAME} \
    --quant_model_path  ${BUILD}/${QUANT_MODEL_FILENAME}
}


train() {
  echo -e "\n-----------------------------------------"
  echo    " TRAINING AND QUANTIZATION PHASE STARTED"
  echo    "-----------------------------------------"

  run_train 2>&1 | tee ${LOG}/${TRAIN_LOG}  

  echo -e "\n-----------------------------------------"
  echo    " TRAINING AND QUANTIZATION COMPLETED"
  echo    "-----------------------------------------"
}

train
