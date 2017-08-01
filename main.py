import tensorflow as tf
import sys
import numpy as np

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
  #hello = tf.constant('Hello, TensorFlow!')
  #sess = tf.Session()
  #print(sess.run(hello))
  argc = len(sys.argv)
  if 5 != argc and 6 != argc:
     print("Error: the number of input parameters is incorrect. Quit.")
     usage()
     return;
  filename = sys.argv[1]
  fileData = open(filename)
  fileBegin = fileData.tell()
  epoches = int(sys.argv[2])
  hiddrenLayerList = sys.argv[3]
  learningRate = float(sys.argv[4])
  totalExamples = sum(1 for line in fileData)
  fileData.seek(fileBegin)
  numTrainTestExamples = 0;
  if 5 == argc:
      numTrainTestExamples = int(totalExamples* 0.8)
  else:
      numTrainTestExamples = int(sys.argv[5])

  # read csv file
  data  = np.ndarray(shape=(totalExamples,2), dtype=int)
  label = np.ndarray(shape=(totalExamples,1), dtype=int)
  row = 0
  for line in fileData:
    rowList = line.split(',')
    rowList.pop(-1)
    rowWidth = len(rowList)
    data[row] = rowList[0:rowWidth-1]
    label[row] = rowList[rowWidth-1]
    row += 1
    if row > 1000: break # for test
  fileData.close()

  #split train and test data and label
  nTrain = int(totalExamples*0.8)
  nTest = totalExamples- nTrain
  trainData = data[0:nTrain]
  trainLabel = label[0:nTrain]
  testData = data[nTrain:totalExamples]
  testLabel = label[nTrain:totalExamples]

  

  print("I love this game")



if __name__ == "__main__":
   main()
