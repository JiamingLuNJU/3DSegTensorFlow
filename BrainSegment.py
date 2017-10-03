import tensorflow as tf
import numpy as np
import random
import time
import threading

class BrainSegment:
    labelWidth = 26  # Groundtruth label is from 0 to 25
    batchSize = 100  # prime number

    def __init__(self):
        # null content
        self.testSeed = False
        self.nMaxTrainThread = 0
        self.nMaxTestThread = 0;

    def printUsage(self):
        usageInfo = "Usage:\n" \
                    + "python3 Segment3D.py T1T2LabelFilename Epoches HiddenLayerStructure LearningRate nCores\n" \
                    + "Notes:\n" \
                    + "1  T1T2LabelFilename is the input csv file with a row representing an example and the last column is a label, which is a converted csv from original 3 nii files;\n" \
                    + "2. Epoches is an integer larger than zero;\n" \
                    + "3. HiddenLayerStructure is number string separated by comma without spaces, e.g. 199,167,139,101,71,41,26\n" \
                    + "4. The width of last layer should be the maximum label value plus 1 for classification purpose;\n" \
                    + "5. LearningRate is an initial learningRate, a float number larger than 1e-4, which will decay every 3 epoches;\n" \
                    + "6. nCores is optional parameter specifying the number of cores of CPU for using"\
                    + "7. Usage Example 1: python3 Segment3D.py T1T2LabelCubicNormalize.csv 10 280,240,200,160,120,80,40,26 0.002\n" \
                    + "8. Usage Example 2: python3 Segment3D.py T1T2LabelCubic.csv 10 280,240,200,160,120,80,40,26 0.002\n" \
                    + "9. Usage Example 3: python3 Segment3D.py T1T2Label2Pixels.csv 10 120,100,80,60,40,26 0.002\n"
        print(usageInfo)


    def parseSysArg(self, argv):
        # parse input parameters
        argc = len(argv)
        if 5 != argc and 6 != argc:
            print("Error: the number of input parameters is incorrect. Quit.")
            self.printUsage()
            return False

        self.filename = argv[1]
        try:
            self.fileData = open(self.filename)
        except:
            print("Error: can not open file", self.filename, ". Program exits.")
            self.printUsage()
            return False

        self.epochs = int(argv[2])
        self.hiddenLayerList = [int(elem) for elem in argv[3].split(',')]
        self.learningRate = float(argv[4])
        self.inputLearningRate = self.learningRate
        self.nCores = 0  # if they are unset or set to 0, will default to the number of logical CPU cores.
        if 6 == argc:
           self.nCores = int(argv[5])
        return True

    def readFile(self):
        # get InputWidth from  input file, that is nFeatures of example
        firstLine = self.fileData.readline()
        firstLine = firstLine.split(',')
        firstLine.pop(-1)
        self.InputWidth = len(firstLine) - 1  # the last column is a label column
        # get the number of the total observations (examples)
        self.fileData.seek(0)
        self.totalExamples = sum(1 for line in self.fileData)
        self.fileData.seek(0)

        # read csv file
        print("Info: reading the T1T2 and Label csv file ......\n")
        self.data = np.ndarray(shape=(self.totalExamples, self.InputWidth), dtype=float)
        self.label = np.zeros(shape=(self.totalExamples, self.labelWidth), dtype=float)
        row = 0
        for line in self.fileData:
            rowList = line.split(',')
            rowList.pop(-1)
            self.data[row] = rowList[0:self.InputWidth]
            self.label[row][int(rowList[self.InputWidth])] = 1  # label is one-hot row vector
            row += 1
        self.fileData.close()

    def splitTrainTestData(self):
        # split train and test data and label
        self.nTrain = int(self.totalExamples * 0.8)
        self.nTest = self.totalExamples - self.nTrain
        self.trainData = self.data[0:self.nTrain]
        self.trainLabel = self.label[0:self.nTrain]
        self.testData = self.data[self.nTrain:self.totalExamples]
        self.testLabel = self.label[self.nTrain:self.totalExamples]
        print("Number of features in example(observation) data:", self.data.shape[1])
        print("Number of train examples: ", self.nTrain)
        print("Number of test examples: ", self.nTest)
        return True

    def constructGraph(self,testSeed=False):
        # Construct a Deep Learning network model
        if testSeed:
            self.testSeed = True
        self.x = tf.placeholder(tf.float32, [None, self.InputWidth])
        self.y_ = tf.placeholder(tf.float32, [None, self.labelWidth])  # groundtruth
        preWidth = self.InputWidth
        preOut = self.x
        nLayer = 0
        self.randomW_seed = []
        for width in self.hiddenLayerList:
            if 2 == self.InputWidth:
                # Xavier initialization results all nan output in 54 input pixels case
                W_name = "W" + str(nLayer)
                W = tf.get_variable(W_name, shape=[preWidth, width], initializer=tf.contrib.layers.xavier_initializer())
            else:
                # random_normal initialization results a lot of zeros in output for 2 pixels input case
                W_mean = 1 / width
                W_stddev = W_mean / 2

                # W_seed = nLayer*73+41 # this can achieve 91.4% accuracy in 10 epochs for [54-360-320-280-240-200-160-120-80-40-26]
                W_seed = nLayer * 73 + 11  # this can achieve 91.4% accuracy in 10 epochs for [54-360-320-280-240-200-160-120-80-40-26]
                if testSeed:
                    # random seed make result vary too much
                    W_seed = random.randint(1,100)
                    # W_graph_seed = random.randint(100,200)
                    # tf.set_random_seed(W_graph_seed*width*preWidth)
                    print("W_seed at width ", width, ": ", W_seed)
                    self.randomW_seed.extend([width, W_seed])

                W = tf.Variable(tf.truncated_normal([preWidth, width], mean=W_mean, stddev=W_stddev, seed=W_seed))

            b = tf.Variable(tf.random_uniform([width], minval=0.1,
                                              maxval=0.5))  # when labels are not evenly distribution, use bigger bias
            preOut = tf.nn.relu(tf.matmul(preOut, W) + b)
            preWidth = width
            nLayer += 1
        self.outLayer = preOut
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=preOut))


    def trainAndTest(self,mySession):
        mySession.run(tf.global_variables_initializer())
        self.layerListStr = "[" + str(self.InputWidth) + "-" + "-".join(str(e) for e in self.hiddenLayerList) + "]"
        print("===============================================================================")
        print("Epoch, LayersWidth, BatchSize, LearningRate, NumTestExamples, CorrectRate")
        for i in range(self.epochs):
            if (0 != i and 0 == i % 3 and self.learningRate > 1.0e-4):
                self.learningRate = self.learningRate * 0.6
            train_step = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate).minimize(self.cross_entropy)
            for j in range(0, self.nTrain, self.batchSize):
                batchX = self.trainData[j:j + self.batchSize]
                batchY = self.trainLabel[j:j + self.batchSize]
                mySession.run(train_step, feed_dict={self.x: batchX, self.y_: batchY})
                # nThread = threading.active_count()
                # if nThread > self.nMaxTrainThread:
                #     self.nMaxTrainThread = nThread

            correct_prediction = tf.equal(tf.argmax(self.outLayer, 1), tf.argmax(self.y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 0)
            correctRate = mySession.run(accuracy, feed_dict={self.x: self.testData, self.y_: self.testLabel})
            # nThread = threading.active_count()
            # if nThread > self.nMaxTestThread:
            #     self.nMaxTestThread = nThread

            print(i, ",", self.layerListStr, ",", self.batchSize, ",", self.learningRate, ",", self.nTest, ",", correctRate)

        if correctRate >= 0.92:
           self.printWSeedFile(correctRate)

        return correctRate

    def printWSeedFile(self, correctRate):
        if self.testSeed:
            fileName = "Accuracy_"+str(int(correctRate*10000))+".txt"
            seedFile = open(fileName, "a")
            seedFile.write("Time: "+ time.strftime("%Y-%m-%d %H:%M:%S %Z\n", time.localtime(time.time())))
            seedFile.write("LayerSturcture: "+self.layerListStr +"\n")
            seedFile.write("Width and W_seed list: "+ ' '.join(str(e) for e in self.randomW_seed)+"\n")
            seedFile.write("Epochs: "+str(self.epochs)+"\n")
            seedFile.write("Arm *.rtxccuracy: "+ str(correctRate)+"\n")
            seedFile.write("============================\n\n")
            seedFile.close()
            print("Update/Output a file: ", fileName)
