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


#---------------------------------SETTING-THE-ENVIROMENT---------------------------------

import keras

from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

#-------------------------------CUSTOM-NETWORK-ARCHITECTURE-------------------------------

def customcnn(h,w,c,num_classes):

	inputs = keras.Input(shape=(h,w,c))

	'''

	#---------------------------CIFAR10-NETWORK-ARCHITECTURE---------------------------

	x = layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(h,w,c))(inputs)
	x = layers.BatchNormalization()(x)
	x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling2D(pool_size=(2,2))(x)
	x = layers.Dropout(0.3)(x)

	x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling2D(pool_size=(2,2))(x)
	x = layers.Dropout(0.5)(x)

	x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling2D(pool_size=(2,2))(x)
	x = layers.Dropout(0.5)(x)

	x = layers.Flatten()(x)
	x = layers.Dense(128, activation='relu')(x)

	#x = layers.BatchNormalization()(x)
 
	x = layers.Dropout(0.5)(x)
	outputs = layers.Dense(num_classes, activation='softmax')(x)

	model = keras.Model(inputs=inputs, outputs=outputs, name='customcnn')

	'''

	#---------------------------MNIST-NETWORK-ARCHITECTURE---------------------------

	x = layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(h,w,c))(inputs)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling2D(pool_size=(2,2))(x)

	x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(inputs)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling2D(pool_size=(2,2))(x)

	x = layers.Flatten()(x)
	outputs = layers.Dense(num_classes, activation='softmax')(x)

	model = keras.Model(inputs=inputs, outputs=outputs, name='customcnn')

	# Checking the model summary
	model.summary()

	return model
