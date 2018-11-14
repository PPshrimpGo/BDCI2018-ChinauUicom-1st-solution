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
  python3 model2.py
  echo '**' `date +%H:%M:%S` 'finished model1'
fi

