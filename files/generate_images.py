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


import os
import argparse
import cv2
import tensorflow as tf

import numpy as np

from tensorflow.keras.datasets import cifar10, mnist


def gen_images(dataset, subset, image_dir, max_images, image_format):

  one_chan = False
  
  # make the calibration images folder if it doesn't exist
  if not os.path.isdir(image_dir):
    os.makedirs(image_dir)

  # Fetch the Keras dataset
  if (dataset=='mnist'):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    classes = ['zero','one','two','three','four','five','six','seven','eight','nine']
    one_chan = True
  else:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

  # which subset?
  if (subset=='train'):
    data_array = x_train
    label_array = y_train
  else:
    data_array = x_test
    label_array = y_test

  # Convert numpy arrays of dataset subset into image files.
  for i in range(len(data_array[:max_images])):

    img_file=os.path.join(image_dir, classes[int(label_array[i])]+'_'+str(i)+'.'+image_format)
    
    if (one_chan == True):
      img = cv2.cvtColor(data_array[i], cv2.COLOR_GRAY2BGR)
    else:
      img = cv2.cvtColor(data_array[i], cv2.COLOR_RGB2BGR)

    # imwrite assumes BGR format
    cv2.imwrite(img_file, img)

  return


# only used if script is run as 'main' from command lime
def main():
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  
  ap.add_argument('-d', '--dataset',
                  type=str,
                  default='mnist',
                  choices=['cifar10','mnist'],
                  help='Dataset - valid choices are cifar10, mnist. Default is mnist') 
  ap.add_argument('-s', '--subset',
                  type=str,
                  default='test',
                  choices=['train','test'],
                  help='Convert training or test subset - valid choices are train, test. Default is test') 
  ap.add_argument('-dir', '--image_dir',
                  type=str,
                  default='image_dir',
                  help='Path to folder for saving images and images list file. Default is image_dir')  
  ap.add_argument('-f', '--image_format',
                  type=str,
                  default='png',
                  choices=['png','jpg','bmp'],
                  help='Image file format - valid choices are png, jpg, bmp. Default is png')  
  ap.add_argument('-m', '--max_images',
                  type=int,
                  default=1000,
                  help='Number of images to generate. Default is 1000')
  
  args = ap.parse_args()  
  
  print ('Command line options:')
  print (' --dataset      : ', args.dataset)
  print (' --subset       : ', args.subset)
  print (' --image_dir    : ', args.image_dir)
  print (' --image_format : ', args.image_format)
  print (' --max_images   : ', args.max_images)


  gen_images(args.dataset, args.subset, args.image_dir, args.max_images, args.image_format)


if __name__ == '__main__':
  main()
