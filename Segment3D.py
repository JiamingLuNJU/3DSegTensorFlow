'''
Performance Comparison between Intel DAAL and Tensorflow neural network

Software packages: Intel DAAL and Tensorflow 1.3;

Author: Hui Xie August 2017

'''
## this is a stable version


import tensorflow as tf
import sys
import numpy as np
import time
import random

LabelWidth = 26  # Groundtruth label is from 0 to 25
batchSize = 100  #prime number

def usage():
  usageInfo = "Usage:\n"\
         + "python3 Segment3D.py T1T2LabelFilename Epoches HiddenLayerStructure LearningRate\n"\
         + "Notes:\n"\
         + "1  T1T2LabelFilename is the input csv file with a row representing an example and the last column is a label, which is a converted csv from original 3 nii files;\n"\
         + "2. Epoches is an integer larger than zero;\n"\
         + "3. HiddenLayerStructure is number string separated by comma without spaces, e.g. 199,167,139,101,71,41,26\n"\
         + "4. The width of last layer should be the maximum label value plus 1 for classification purpose;\n"\
         + "5. LearningRate is an initial learningRate, a float number larger than 1e-4, which will decay every 3 epoches;\n"\
         + "6. Usage Example: python3 Segment3D.py T1T2LabelCubic.csv 10 240,200,160,120,80,40,26 0.002\n"
  print(usageInfo)

def main():
  # parse input parameters
  argc = len(sys.argv)
  if 5 != argc:
     print("Error: the number of input parameters is incorrect. Quit.")
     usage()
     return

  filename = sys.argv[1]
  try:
    fileData = open(filename)
  except:
    print("Error: can not open file", filename, ". Program exits.")
    usage()
    return

  epoches = int(sys.argv[2])
  hiddenLayerList = [int(elem) for elem in sys.argv[3].split(',')]
  learningRate = float(sys.argv[4])

  # get InputWidth from  input file, that is nFeatures of example
  firstLine = fileData.readline()
  firstLine = firstLine.split(',')
  firstLine.pop(-1)
  InputWidth = len(firstLine) -1 # the last column is a label column
  # get the number of the total observations (examples)
  fileData.seek(0)
  totalExamples = sum(1 for line in fileData)
  fileData.seek(0)

  # read csv file
  print("Info: reading the T1T2 and Label csv file ......\n")
  data  = np.ndarray(shape=(totalExamples,InputWidth), dtype=int)
  label = np.zeros(shape=(totalExamples,LabelWidth), dtype=int)
  row = 0
  for line in fileData:
    rowList = line.split(',')
    rowList.pop(-1)
    data[row] = rowList[0:InputWidth]
    label[row][int(rowList[InputWidth])] = 1  # label is one-hot row vector
    row += 1
  fileData.close()

  #split train and test data and label
  nTrain = int(totalExamples*0.8)
  nTest = totalExamples- nTrain
  trainData = data[0:nTrain]
  trainLabel = label[0:nTrain]
  testData = data[nTrain:totalExamples]
  testLabel = label[nTrain:totalExamples]
  print("Number of features in example(observation) data:", data.shape[1])
  print("Number of train examples: ", nTrain)
  print("Number of test examples: ", nTest)

  #===============Debug=============================
  #nZeroAtFirstCol = 0;
  #for i in range(nTest):
  #  if testLabel[i,0] == 1:
  #      nZeroAtFirstCol +=1
  #print("Rate of 1s at first column of label,which means network output all zeros, all nan, or always maximum at first column, will get this result): ", nZeroAtFirstCol/nTest)
  # 0.5247328947864316 for cubic case, 0.53711 for 2 pixel case.
  #===============Debug==============================

  # start time computation
  print("Start Tensorflow Neural Network at:", time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(time.time())))
  startTime = time.perf_counter()

  # Construct a Deep Learning network model
  x = tf.placeholder(tf.float32,[None,InputWidth])
  y_ = tf.placeholder(tf.float32,[None,LabelWidth])  # groundtruth
  preWidth = InputWidth
  preOut = x
  nLayer = 0
  for width in hiddenLayerList:
    if 2 == InputWidth:
       # Xavier initialization results all nan output in 54 input pixels case
       W_name = "W"+str(nLayer)
       W = tf.get_variable(W_name, shape=[preWidth,width], initializer=tf.contrib.layers.xavier_initializer())
    else:
       # random_normal initialization results a lot of zeros in output for 2 pixels input case
       W_mean = 1/width
       W_stddev = W_mean/2
       W_seed = random.randint(7,127)
       W_graph_seed = random.randint(11,197)
       tf.set_random_seed(W_graph_seed*width*preWidth)
       W = tf.Variable(tf.truncated_normal([preWidth, width], mean=W_mean, stddev=W_stddev, seed=W_seed))

    b = tf.Variable(tf.random_uniform([width],minval=0.1,maxval=0.5))
    preOut = tf.nn.relu(tf.matmul(preOut, W) + b)
    preWidth = width
    nLayer += 1

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=preOut))

  # train and test
  mySession = tf.Session()
  mySession.run(tf.global_variables_initializer())
  layerListStr = "["+str(InputWidth)+"-"+ "-".join(str(e) for e in hiddenLayerList)+"]"
  print("===============================================================================")
  print("Epoch, LayersWidth, BatchSize, LearningRate, NumTestExamples, CorrectRate")
  for i in range(epoches):
     if (0 != i and 0 == i % 3 and learningRate > 1.0e-8):
       learningRate = learningRate * 0.6
     train_step = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(cross_entropy)
     for j in range(0, nTrain, batchSize):
        batchX = trainData[j:j+batchSize]
        batchY = trainLabel[j:j+batchSize]
        mySession.run(train_step, feed_dict={x:batchX, y_:batchY})

     correct_prediction = tf.equal(tf.argmax(preOut, 1), tf.argmax(y_, 1))
     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 0)
     correctRate = mySession.run(accuracy, feed_dict={x:testData, y_:testLabel})
     print(i,",",layerListStr,",",batchSize,",",learningRate,",", nTest,",",correctRate)

     #============Debug =======================
     #testOut = mySession.run(preOut, feed_dict={x:testData, y_:testLabel})
     #print("shape of preOut:", testOut.shape)
     #print(testOut)
     #==============Debug=======================

  mySession.close()

  diffTime = time.perf_counter() - startTime
  print("==========End of Tensorflow Neural Network=============")
  print("Computation time for Tensorflow Neural Network: ", diffTime, "seconds.")
  print("End Tensorflow Neural Network at:", time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(time.time())))

if __name__ == "__main__":
   main()
