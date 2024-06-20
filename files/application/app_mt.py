from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse

def preprocess_fn(image_path, fix_scale):
    '''
    Image pre-processing.
    Opens image as color, adds channel dimension, normalizes to range 0:1
    and then scales by input quantization scaling factor
    input arg: path of image file
    return: numpy array
    '''

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    #MNIST IMAGE TYPE
    image = image.reshape(28,28,1)                          #MNIST IMAGE SHAPE

    #image = cv2.imread(image_path, cv2.IMREAD_COLOR)       #CIFAR10 IMAGE TYPE
    #image = image.reshape(32,32,3)                         #CIFAR10 IMAGE SHAPE

    '''normalization and scaling'''
    image = image * (1/255.0) * fix_scale

    '''convert to int8'''
    image = image.astype(np.int8)

    return image


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:

    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def runDPU(id,start,dpu,img):

    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    # we can avoid output scaling if use argmax instead of softmax
    #output_fixpos = outputTensors[0].get_attr("fix_point")
    #output_scale = 1 / (2**output_fixpos)

    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    write_index = start
    ids=[]
    ids_max = 10
    outputData = []

    for i in range(ids_max):
        outputData.append([np.empty(output_ndim, dtype=np.int8, order="C")])

    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count

        '''prepare batch input/output '''
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])

        '''run with batch '''
        job_id = dpu.execute_async(inputData,outputData[len(ids)])
        ids.append((job_id,runSize,start+count))
        count = count + runSize 
        if count<n_of_images:
            if len(ids) < ids_max-1:
                continue
        for index in range(len(ids)):
            dpu.wait(ids[index][0])
            write_index = ids[index][2]
            '''store output vectors '''
            for j in range(ids[index][1]):
                # we can avoid output scaling if use argmax instead of softmax
                # out_q[write_index] = np.argmax(outputData[0][j] * output_scale)
                out_q[write_index] = np.argmax(outputData[index][0][j])
                write_index += 1
        ids=[]

#-------------------------------------MAIN-SCRIPT-------------------------------------

def app(image_dir,threads,model):

    #-----------------------------PRELIMINARY-OPERATIONS------------------------------

    '''get images and how much they are'''
    listimage=os.listdir(image_dir)
    runTotal = len(listimage)

    '''global predictions array'''
    global out_q
    out_q = [None] * runTotal

    '''get graph and subgraph'''
    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)

    '''create dpu runners based on number of threads'''
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    '''compute input scaling factor'''
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos


    #--------------------------------PREPROCESS-IMAGES--------------------------------

    print("--------------------------------------------")
    print("\n START PREPROCESS ", runTotal, " IMAGES...")

    img = []
    for i in range(runTotal):
        path = os.path.join(image_dir,listimage[i])
        img.append(preprocess_fn(path, input_scale))

    print("\n IMAGE PREPROCESSING ENDED")


    #-----------------------------------RUN-THREADS-----------------------------------

    print("\n--------------------------------------------")
    print("\n STARTING ", threads, " THREADS...")

    threadAll = []
    start=0
    for i in range(threads):
        if (i==threads-1):
            end = len(img)
        else:
            end = start+(len(img)//threads)
        in_q = img[start:end]
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start=end

    print("\n START INFERENCE...")

    time1 = time.time()

    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()

    time2 = time.time()
    
    timetotal = time2 - time1

    print("\n INFERENCE ENDED")


    #--------------------------------PROCESS-RESULTS--------------------------------

    print("\n--------------------------------------------")
    print("\n START RESULTS PROCESSING...")

    '''Througput'''
    fps = float(runTotal / timetotal)


    '''Check matches between predicted and true labels'''
    classes = ['zero','one','two','three','four','five','six','seven','eight','nine']                  #MNIST CLASSES
    #classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']	    #CIFAR10 CLASSES

    correct = 0
    wrong = 0

    for i in range(len(out_q)):
        prediction = classes[out_q[i]]
        ground_truth, _ = listimage[i].split('_',1)

        if (ground_truth==prediction):
            correct += 1
        else:
            wrong += 1
    accuracy = correct/len(out_q)

    print("\n RESULTS PROCESSING ENDED")


    #------------------------------------RESULTS------------------------------------

    print("\n--------------------------------------------")
    print("\n                 RESULTS")                 

    print("\n\n Throughput=%.2f fps, total frames = %.0f, time=%.6f seconds" %(fps, runTotal, timetotal))

    print('\n\n Correct:%d, Wrong:%d, Accuracy:%.2f' %(correct,wrong,accuracy))

    print("\n--------------------------------------------")

    return


#---------------------------------------MAIN----------------------------------------

''' only used if script is run as 'main' from command line '''
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()  
  ap.add_argument('-d', '--image_dir', type=str, default='images',                      help='Path to folder of images. Default is images')  
  ap.add_argument('-t', '--threads',   type=int, default=1,                             help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',     type=str, default='model_dir/customcnn.xmodel',  help='Path of xmodel. Default is model_dir/customcnn.xmodel')

  args = ap.parse_args()  
  
  print ('Command line options:')
  print (' --image_dir : ', args.image_dir)
  print (' --threads   : ', args.threads)
  print (' --model     : ', args.model)

  app(args.image_dir,args.threads,args.model)

if __name__ == '__main__':
  main()
