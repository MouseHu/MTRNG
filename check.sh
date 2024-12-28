#!/bin/sh
python train.py
ret=$?
if [ $ret -eq 0 ]; then
  echo "The program exited normally";
elif [ $ret -gt 128 ]; then
  echo "The program died of signal $((ret-128)): $(kill -l $ret)"
else
  echo "The program failed with status $ret"
fi