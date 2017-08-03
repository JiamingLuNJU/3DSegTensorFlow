'''
Performance Comparison between Intel DAAL and Tensorflow neural network

Software packages: Intel DAAL and Tensorflow 1.3;

Author: Hui Xie August 2017

'''
import tensorflow as tf
import sys
import numpy as np
import time

InputWidth = 2
LabelWidth = 26  # Groundtruth label is from 0 to 25
batchSize = 100

#visualGraph = True

def usage():
  usageInfo = "Usage:\n"\
         + "python3 main.py T1T2LabelFilename Epoches HiddenLayerStructure LearningRate\n"\
         + "Notes:\n"\
         + "1  T1T2LabelFilename is the input csv file with a row representing an example and the last column is a label, which is a converted form original nii files;\n"\
         + "2. Epoches is an integer larger than zero;\n"\
         + "3. HiddenLayerStructure is number string separated by comma without a space. e.g. 80,60,50,50,26\n"\
         + "4. The width of last layer should be the maximum label value plus 1 for classification goal;\n"\
         + "5. LearningRate is an initial learningRate, a float number larger than 1e-4, which will decay every 3 epoches;\n"\
         + "6. Usage Example: python3 ./main.py T1T2Label.csv 20  80,60,40,26 0.002\n"
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
  fileData.seek(0)
  totalExamples = sum(1 for line in fileData)
  fileData.seek(0)

  # read csv file
  print("Infor: reading the T1T2 and Label csv file ......")
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

  #debug
  nRow = testLabel.shape[0]
  rate = testLabel[:,0].sum()
  print("testLabel 1 rate at first column IS 0.537711.=====", rate*1.0/nRow)
  print("!!!!!This is the problem!!!!!")



  # start time computation
  print("Start Tensorflow Neural Network at:", time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(time.time())))
  startTime = time.perf_counter()

  # Consturct a Deep Learning network model
  x = tf.placeholder(tf.float32,[None,InputWidth])
  yGroundTruth = tf.placeholder(tf.float32,[None,LabelWidth])
  preWidth = InputWidth
  preOut = x
  for width in hiddenLayerList:
    W = tf.Variable(tf.zeros([preWidth,width]))
    b = tf.Variable(tf.zeros([width]))
    preOut = tf.nn.relu(tf.matmul(preOut, W) + b)
    preWidth = width

  #####Use another method to construct network
  # WList =[]
  # bList = []
  # layerResultlList = [x]
  # sizeHiddenLayers = len(hiddenLayerList)
  # for i in range(sizeHiddenLayers):
  #     width = hiddenLayerList[i]
  #     WList.append( tf.Variable(tf.zeros([preWidth,width])) )
  #     bList.append( tf.Variable(tf.zeros([width])) )
  #     layerResultlList.append( tf.nn.relu(tf.matmul(layerResultlList[i], WList[i]) + bList[i]) )
  #     preWidth = width

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yGroundTruth,logits=preOut))

  # train and test
  mySession = tf.Session()
  mySession.run(tf.global_variables_initializer())
  hiddenLayerListStr = "["+ "-".join(str(e) for e in hiddenLayerList)+"]"
  print("Total train examples: ", nTrain)
  print("===============================================================================")
  print("Epoch, HiddenLayersWidth, BatchSize, LearningRate, NumTestExamples, CorrectRate")
  for i in range(epoches):
     train_step = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(cross_entropy)
     if (0 != i and 0 == i % 3 and learningRate > 1.0e-8):
       learningRate = learningRate * 0.7

     for j in range(0, nTrain, batchSize):
        batchX = trainData[j:j+batchSize]
        batchY = trainLabel[j:j+batchSize]
        mySession.run(train_step, feed_dict={x:batchX, yGroundTruth:batchY})

     # test in every epoch
     correct_prediction = tf.equal(tf.argmax(preOut, 1), tf.argmax(yGroundTruth, 1))
     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 0)
     correctRate = mySession.run(accuracy, feed_dict={x:testData, yGroundTruth:testLabel})
     print(i,",",hiddenLayerListStr,",",batchSize,",",learningRate,",", nTest,",",correctRate)
  mySession.close()

  diffTime = time.perf_counter()-startTime
  print("End Tensorflow Neural Network at:", time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(time.time())))
  print("Computation time for Tensorflow Neural Network: ", diffTime, "seconds.")
  print("==========End of Tensorflow Neural Network=============")

if __name__ == "__main__":
   main()
