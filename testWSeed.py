# This a version for Tensorflow Performance test in W seed

import BrainSegment
import sys
import time

def main():
   print ("$$$$$$$$$$$ test random W seed **********")
   brainSegment= BrainSegment.BrainSegment()
   if False == brainSegment.parseSysArg(sys.argv):
       return
   brainSegment.readFile()
   brainSegment.splitTrainTestData()

   nCount = 60
   for i in range(nCount):
       print ("\n****   Current test iteration: ",i, "*******")
       mySession = BrainSegment.tf.Session()
       startTime = time.perf_counter()
       brainSegment.constructGraph(testSeed=True)
       brainSegment.learningRate = brainSegment.inputLearningRate
       brainSegment.trainAndTest(mySession)
       mySession.close()
       diffTime = time.perf_counter() - startTime
       print("Computation time for", i, "cores: ",diffTime, "seconds.")

if __name__ == "__main__":
   main()