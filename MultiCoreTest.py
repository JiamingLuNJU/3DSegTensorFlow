# This a version for Tensorflow Performance test in MultiCore

import BrainSegment
import sys
import time

def main():
   print ("$$$$$$$$$$$ MultiCore Performance test**********")
   brainSegment= BrainSegment.BrainSegment()
   if False == brainSegment.parseSysArg(sys.argv):
       return
   brainSegment.readFile()
   brainSegment.splitTrainTestData()

   nCores = 8+1
   for i in range(1,nCores,1):
       print ("\n\n$$$$   Current use core: ",i, "$$$$")
       config = BrainSegment.tf.ConfigProto(
           device_count={'CPU': i},
           intra_op_parallelism_threads=i*2,
           inter_op_parallelism_threads=i*2*2,
           use_per_session_threads=True)
       mySession = BrainSegment.tf.Session(config=config)
       #print("Start Tensorflow Neural Network at:", time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(time.time())))
       startTime = time.perf_counter()

       brainSegment.constructGraph()

       brainSegment.trainAndTest(mySession)
       mySession.close()

       diffTime = time.perf_counter() - startTime
       #print("==========End of Tensorflow Neural Network=============")
       print("Computation time for", i, "cores: ",diffTime, "seconds.")
       #print("End Tensorflow Neural Network at:", time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(time.time())))

if __name__ == "__main__":
   main()