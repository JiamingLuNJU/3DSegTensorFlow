#!/bin/bash
# Python3 Segmentation Script run in Putamen

for i in $(seq 16)
do
  if (($i == 1 || $i %2 ==0))
  then
     echo $i
     echo "%%%%%%%%%%%%%%%%%%%%%%%% Start Segment in $i cores %%%%%%%%%%%%%%%%%%%%%"
     python3 ~/Projects/3DSegTensorFlow/MultiCoreTest.py ~/temp/T1T2LabelCubicNormalize.csv 10 240,200,160,120,80,40,26 0.002 $i
     echo "%%%%%%%%%%%%%%%%%%%%%%%% End Segment in $i cores %%%%%%%%%%%%%%%%%%%%%"
  fi
done