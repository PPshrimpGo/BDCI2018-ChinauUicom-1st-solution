#!/bin/bash

stage=0

echo '**' `date +%H:%M:%S` 'start with stage=' $stage 
echo 'Any question contact QQ:674785731'

mkdir -p cv
mkdir -p sub

# gen magic feature
if [ $stage -le 0 ]; then
  cd feature && python3 get_most.py && cd ..
  echo '**' `date +%H:%M:%S` 'finished get most'
fi

# white
if [ $stage -le 1 ]; then
  cd feature && python3 white.py && cd ..
  echo '**' `date +%H:%M:%S` 'finished white'
fi
#model 2
if [ $stage -le 2 ]; then
  python3 model1.py
  echo '**' `date +%H:%M:%S` 'finished jiajie model1'
fi

if [ $stage -le 3 ]; then
  python3 fast_baseline_v11.py
  echo '**' `date +%H:%M:%S` 'finished final'
fi

if [ $stage -le 4 ]; then
  python3 piupiu_white.py
  echo '**' `date +%H:%M:%S` 'finished piupiuwhite,sub_final_while'
fi

