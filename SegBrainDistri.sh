#!/bin/bash
# Python3 Segmentation Distributed Script runned in Argon
qsub -b y -cwd -N SegBrain -q HJ,COE,UI -pe smp 168  -e ~/temp/segError.txt -o ~/temp/segStdOutput.txt \
     ~/intel/intelpython3/bin/python3 ~/Projects/3DSegTensorFlow/MultiCoreTest.py ~/temp/T1T2LabelCubicNormalize.csv 3 240,200,160,120,80,40,26 0.002 10

