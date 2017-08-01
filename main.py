import tensorflow as tf
import sys
import numpy as np

InputWidth = 2
LabelWidth = 26

def usage():
  usageInfo = "Usage:\n"\
         + "command T1T2LabelFilename Epoches HiddenLayerStructure LearningRate [NumberofTrainExample]\n"\
         + "Notes:\n"\
         + "1 T1T2LabelFilename is the input csv file with a row representing an example and the last column is a label;\n"\
         + "2. Epoches is an integer larger than zero;\n"\
         + "3. HiddenLayerStructure is number string separated by comma without a space. e.g. 80,60,50,50,26\n"\
         + "4. The width of last layer should be the maximum label value plus 1 for classification;\n"\
         + "5. LearningRate is an initial learningRate, a float number larger than 1e-4;\n"\
         + "6. NumberofTrainExample is an optional parameter to specify the number of train Example limited to less than 2 million; Without it, program will train all examples.\n"\
         + "7. Example: ./main.py T1T2Label.csv 20  80,60,40,26 0.002 100000\n"
  print(usageInfo)

def main():
  argc = len(sys.argv)
  if 5 != argc and 6 != argc:
     print("Error: the number of input parameters is incorrect. Quit.")
     usage()
     return;
  filename = sys.argv[1]
  fileData = open(filename)
  epoches = int(sys.argv[2])
  hiddrenLayerList = [int(elem) for elem in sys.argv[3].split(',')]
  learningRate = float(sys.argv[4])
  totalExamples = sum(1 for line in fileData)
  fileData.seek(0)
  numTrainTestExamples = 0;
  if 5 == argc:
      numTrainTestExamples = int(totalExamples* 0.8)
  else:
      numTrainTestExamples = int(sys.argv[5])

  # read csv file
  #totalExamples = 10000 ##################################### for test
  data  = np.ndarray(shape=(totalExamples,InputWidth), dtype=int)
  label = np.zeros(shape=(totalExamples,LabelWidth), dtype=int)
  row = 0
  for line in fileData:
    rowList = line.split(',')
    rowList.pop(-1)
    data[row] = rowList[0:InputWidth]
    label[row][int(rowList[InputWidth])] = 1
    row += 1
    #if row >= totalExamples: break ##################################### for test
  fileData.close()

  #split train and test data and label
  nTrain = int(totalExamples*0.8)
  nTest = totalExamples- nTrain
  trainData = data[0:nTrain]
  trainLabel = label[0:nTrain]
  testData = data[nTrain:totalExamples]
  testLabel = label[nTrain:totalExamples]

  #Consturct Deep Learning network model
  x = tf.placeholder(tf.float32,[None,InputWidth])
  yGroundTruth = tf.placeholder(tf.int16,[None,LabelWidth])
  preWidth = InputWidth
  preX = x
  for width in hiddrenLayerList:
    W = tf.Variable(tf.zeros([preWidth,width]))
    b = tf.Variable(tf.zeros([width]))
    preX = tf.nn.relu(tf.matmul(preX, W) + b)
    preWidth = width
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yGroundTruth,logits=preX))

  # train and test preparation
  train_step = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(cross_entropy)


  # train
  session = tf.Session()
  session.run(tf.global_variables_initializer())
  batchSize = 100

  print("Epoch, HiddenLayersWidth, BatchSize, LearningRate, NumTestExamples, CorrectRate")
  for i in range(epoches):
     for j in range(0, nTrain, batchSize):
        batchX = trainData[j:j+batchSize]
        batchY = trainLabel[j:j+batchSize]
        session.run(train_step, feed_dict={x:batchX, yGroundTruth:batchY})
     # test in every epoch
     correct_prediction = tf.equal(tf.argmax(preX, 1), tf.argmax(yGroundTruth, 1))
     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     correctRate = session.run(accuracy, feed_dict={x:testData, yGroundTruth:testLabel})
     print(i,",",hiddrenLayerList,",",batchSize,",",learningRate,",", nTest,",",correctRate)

  session.close()
  print("==========End of Tensorflow Neural Network=============")

if __name__ == "__main__":
   main()
