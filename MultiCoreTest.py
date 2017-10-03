# This a version for Tensorflow Performance test in MultiCore

import BrainSegment
import sys
import time
import socket

def main():
   brainSegment= BrainSegment.BrainSegment()
   if False == brainSegment.parseSysArg(sys.argv):
       return
   brainSegment.readFile()
   brainSegment.splitTrainTestData()

   print ("\n$$$$   Current use core: ",brainSegment.nCores, "$$$$")
   config = BrainSegment.tf.ConfigProto(
        device_count={'CPU': 0}, # device_count limits the number of CPUs being used, not the number of cores or threads.
                                 # By default all CPUs available to the process are aggregated under cpu:0 device.
        intra_op_parallelism_threads=brainSegment.nCores,
        inter_op_parallelism_threads=brainSegment.nCores,
        use_per_session_threads=True)
   mySession = BrainSegment.tf.Session(config=config)
   #print("Start Tensorflow Neural Network at:", time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(time.time())))
   #mySession = BrainSegment.tf.Session();
   startTime = time.perf_counter()

   brainSegment.constructGraph()

   brainSegment.learningRate = brainSegment.inputLearningRate
   brainSegment.trainAndTest(mySession)
   mySession.close()

   diffTime = time.perf_counter() - startTime
   #print("==========End of Tensorflow Neural Network=============")
   print("Computation time: ", diffTime, "seconds.")
   # print("Maximum train thread number: ", brainSegment.nMaxTrainThread);
   # print("Maximum test thread number: ", brainSegment.nMaxTestThread);
   #print("End Tensorflow Neural Network at:", time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(time.time())))

if __name__ == "__main__":
   main()