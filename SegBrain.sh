#!/bin/bash
# Python3 Segmentation Script runned in Argon
# python3 ~/Projects/3DSegTensorFlow/testBrainSegmentation.py ~/temp/T1T2LabelCubicNormalize.csv 10 120,80,40,26 0.002
for i in `seq 1 60`
do
  if (($i == 1 || $i %4 ==0))
  then
     echo $i
     qsub -b y -cwd -N SegBrain_$i -q HJ,COE,UI -pe smp $i  -e ~/temp/segError.txt -o ~/temp/segStdOutput.txt -M sheenxh@gmail.com  -m ea \
     ~/intel/intelpython3/bin/python3 ~/Projects/3DSegTensorFlow/MultiCoreTest.py ~/temp/T1T2LabelCubicNormalize.csv 20 240,200,160,120,80,40,26 0.002 $i
  fi
done
