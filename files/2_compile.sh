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
ARCH_KV260=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
ARCH_ZCU102=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json

#compile for kria KV260, comment this function if you don't need the kria support to dont't waste time
compile() {
  vai_c_tensorflow2 \
    --model       ${BUILD}/${QUANT_MODEL_FILENAME} \
    --arch        ${ARCH_KV260} \
    --output_dir  ${COMPILE_KV260} \
    --net_name    ${NET_NAME}
}

#compile for ZCU102,  comment this function if you don't need the zcu support to dont't waste time
compile() {
  vai_c_tensorflow2 \
    --model       ${BUILD}/${QUANT_MODEL_FILENAME} \
    --arch        ${ARCH_ZCU102} \
    --output_dir  ${COMPILE_ZCU102} \
    --net_name    ${NET_NAME}
}

echo -e "\n-----------------------------------------"
echo    " COMPILING PHASE STARTED.."
echo    "-----------------------------------------"

#compile for kria KV260, comment this function if you don't need the kria support to dont't waste time
rm -rf ${COMPILE_KV260}
mkdir -p ${COMPILE_KV260}
compile 2>&1 | tee ${LOG}/${COMP_LOG_KV260}

#compile for ZCU102,  comment this function if you don't need the zcu support to dont't waste time
rm -rf ${COMPILE_ZCU102}
mkdir -p ${COMPILE_ZCU102}
compile 2>&1 | tee ${LOG}/${COMP_LOG_ZCU102}

echo -e "\n-----------------------------------------"
echo    " COMPILING PHASE COMPLETED"
echo    "-----------------------------------------"
