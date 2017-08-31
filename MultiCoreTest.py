# This a version for Tensorflow Performance test in MultiCore

import BrainSegment
import sys
import time
import socket

def main():
   print ("$$$$$$$$$$$ MultiCore Performance test**********")
   brainSegment= BrainSegment.BrainSegment()
   if False == brainSegment.parseSysArg(sys.argv):
       return
   brainSegment.readFile()
   brainSegment.splitTrainTestData()

   print ("\n$$$$   Current use core: ",brainSegment.nCores, "$$$$")
   config = BrainSegment.tf.ConfigProto(
       device_count={'CPU': 1}, # device_count limits the number of CPUs being used, not the number of cores or threads.
       intra_op_parallelism_threads=brainSegment.nCores,
       inter_op_parallelism_threads=brainSegment.nCores,
       use_per_session_threads=True)
   mySession = BrainSegment.tf.Session(config=config)
   #print("Start Tensorflow Neural Network at:", time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(time.time())))
   startTime = time.perf_counter()

   brainSegment.constructGraph()

   brainSegment.learningRate = brainSegment.inputLearningRate
   brainSegment.trainAndTest(mySession)
   mySession.close()

   diffTime = time.perf_counter() - startTime
   #print("==========End of Tensorflow Neural Network=============")
   print("Computation time for", brainSegment.nCores, "cores: ",diffTime, "seconds.")
   #print("End Tensorflow Neural Network at:", time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(time.time())))

if __name__ == "__main__":
   print("\n\n\nMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM\n")
   print("====This is multi cores performance test====")
   main()