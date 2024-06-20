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
import sys
import argparse

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import keras

from tensorflow_model_optimization.quantization.keras import vitis_quantize

from customcnn import customcnn
from datadownload import datadownload


#---------------------------------------------------------TRAIN---------------------------------------------------------

def train(input_height, input_width, input_chan, epochs, batchsize, calib_images, float_model_path, quant_model_path):

    #--------------------------PRELIMINARY-OPERATIONS--------------------------

    # dataset download and preparation
    (x_train, y_train), (x_test, y_test) = datadownload()    


    # calculate total number of batches per epoch
    total_batches = int(len(x_train)/batchsize)
    
    #----------------------CREATE-THE-COMPUTATIONAL-GRAPH----------------------

    # Building custom convolutional neural network
    print("\n----------------------------",flush=True)
    print(" BUILDING CUSTOM CNN...",flush=True)
    print("----------------------------\n",flush=True)

    model = customcnn(h = input_height, w = input_width, c = input_chan, num_classes = 10)

    #------------------------COMPILE-AND-FIT-THE-MODEL-------------------------

        # Training phase with training data
    print("\n----------------------------",flush=True)
    print(" TRAINING STARTED...",flush=True)
    print("----------------------------\n",flush=True)

    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=batchsize, epochs=epochs, validation_data=(x_test, y_test))

    #------------------------------SAVE-THE-MODEL------------------------------

    # Save the parameters of the trained network
    print("\n----------------------------",flush=True)
    print(" SAVING FLOAT MODEL...",flush=True)
    print("----------------------------",flush=True)

    model.save(float_model_path)

    #-------------------------------QUANTIZATION-------------------------------

    # Post-Training Quantize
    print("\n----------------------------",flush=True)
    print(" QUANTIZATION STARTED...",flush=True)
    print("----------------------------\n",flush=True)
    
    quantizer = vitis_quantize.VitisQuantizer(model, quantize_strategy='pof2s')

    quantized_model = quantizer.quantize_model(calib_dataset=x_test[0:calib_images], calib_steps=100, calib_batch_size=10)

    #---------------------------SAVE-QUANTIZED-MODEL---------------------------

    # Save Quantized Model
    print("\n----------------------------",flush=True)
    print(" SAVING QUANTIZED MODEL...",flush=True)
    print("----------------------------\n",flush=True)
    
    quantized_model.save(quant_model_path)

    #-------------------------EVALUATE-QUANTIZED-MODEL-------------------------

    # Evaluate Quantized Model
    print("\n----------------------------",flush=True)
    print(" EVALUATING QUANTIZED MODEL...",flush=True)
    print("----------------------------\n",flush=True)

    # Load Quantized Model
    quantized_model = keras.models.load_model(quant_model_path)

    # Evaluate Quantized Model
    quantized_model.compile(loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    quantized_model.evaluate(x_test, y_test, batch_size=1000)

    #--------------------------------------------------------------------------

    return

#---------------------------------------------------------MAIN----------------------------------------------------------

def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument('-ih', '--input_height',
                    type=int,
                    default=32,
                    help='Input data height. Default is 32')
    ap.add_argument('-iw', '--input_width',
                    type=int,
                    default=32,
                    help='Input data width. Default is 32')                  
    ap.add_argument('-ic', '--input_chan',
                    type=int,
                    default=3,
                    help='Input data channels. Default is 3')
    ap.add_argument('-e', '--epochs',
                    type=int,
                    default=100,
                    help='Number of training epochs. Default is 100')
    ap.add_argument('-b', '--batchsize',
                    type=int,
                    default=64,
                    help='Training batchsize. Default is 64')
    ap.add_argument('-ci', '--calib_images',
                    type=int,
                    default=1000,
                    help='Number of images to generate. Default is 1000')
    ap.add_argument('-g', '--gpu',
                    type=str,
                    default='0',
                    help='IDs of GPU cards to be used. Default is 0')
    ap.add_argument('-fm', '--float_model_path',
                    type=str,
                    default='./build/float.h5',
                    help='Path and filename of floating point model. Default is ./build/float.h5')
    ap.add_argument('-qm', '--quant_model_path',
                    type=str,
                    default='./build/quantized.h5',
                    help='Path and filename of quantized model. Default is ./build/quantized.h5')
    args = ap.parse_args() 


    print('\n------------------------------------')
    print('Keras version      :',tf.keras.__version__)
    print('TensorFlow version :',tf.__version__)
    print('Python version     :',(sys.version))
    print('------------------------------------')
    print ('Command line options:')
    print (' --input_height    : ', args.input_height)
    print (' --input_width     : ', args.input_width)
    print (' --input_chan      : ', args.input_chan)
    print (' --epochs          : ', args.epochs)
    print (' --batchsize       : ', args.batchsize)
    print (' --calib_images    : ', args.calib_images)
    print (' --gpu             : ', args.gpu)
    print (' --float_model_path: ', args.float_model_path)
    print (' --quant_model_path: ', args.quant_model_path)
    print('------------------------------------\n')



    os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
    
    # indicate which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


    train(args.input_height,args.input_width,args.input_chan,args.epochs,args.batchsize,args.calib_images,
          args.float_model_path,args.quant_model_path)


if __name__ == '__main__':
  main()
