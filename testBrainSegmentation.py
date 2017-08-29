# this program test the class BrainSegment

import BrainSegment
import sys
import time

def main():
   brainSegment= BrainSegment.BrainSegment()
   if False == brainSegment.parseSysArg(sys.argv):
       return
   brainSegment.readFile()
   brainSegment.splitTrainTestData()

   print("Start Tensorflow Neural Network at:", time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(time.time())))
   startTime = time.perf_counter()

   brainSegment.constructGraph()
   mySession = BrainSegment.tf.Session()
   brainSegment.trainAndTest(mySession)
   mySession.close()

   diffTime = time.perf_counter() - startTime
   print("==========End of Tensorflow Neural Network=============")
   print("Computation time for Tensorflow Neural Network: ", diffTime, "seconds.")
   print("End Tensorflow Neural Network at:", time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(time.time())))

if __name__ == "__main__":
   main()